import streamlit as st
import os
import tempfile
from vectorstore import vectorstore_pipeline, load_vectorstore
from model_configuration import load_json, save_json
from langgraph.checkpoint.memory import MemorySaver

# File paths
AVAILABLE_MODELS_FILE = "available_models.json"
CONFIG_FILE = "config.json"

# Load available models and config
available_models = load_json(AVAILABLE_MODELS_FILE)
config = load_json(CONFIG_FILE)

# Get model options
llm_models = list(available_models.get("llm_models", {}).keys())
embeddings_models = list(available_models.get("embeddings_models", {}).keys())
parsing_methods = ["local", "llama-index"]

# Initialize session state
if 'llm_model' not in st.session_state or 'embeddings_model' not in st.session_state:
    llm_model_from_config = config.get("llm_model", {})
    embeddings_model_from_config = config.get("embeddings_model", {})
    parsing_method_from_config = config.get("parsing_method", {})
    
    if llm_model_from_config and embeddings_model_from_config and parsing_method_from_config:
        st.session_state.llm_model = list(llm_model_from_config.keys())[0]
        st.session_state.embeddings_model = list(embeddings_model_from_config.keys())[0]
        st.session_state.parsing_method = parsing_method_from_config
    else:
        st.session_state.llm_model = llm_models[0] if llm_models else None
        st.session_state.embeddings_model = embeddings_models[0] if embeddings_models else None
        st.session_state.parsing_method = parsing_methods[1]

if 'vectorstore_created' not in st.session_state:
    st.session_state.vectorstore_created = False

if 'memory' not in st.session_state:
    st.session_state.memory = MemorySaver()


# Sidebar UI
with st.sidebar:
    st.title("💬 Local RAG")
    st.markdown("""Your document-based RAG chatbot. Select models, upload PDFs, and start chatting.""")

    if st.button("🗑️ Clear Conversation History"):
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.session_state.memory.clear()  # Clear memory saved in MemorySaver
        st.success("🗑️ Conversation history cleared!")

    vectorstore_action = st.radio("Choose an action", ("Create Vectorstore", "Load Existing Vectorstore"))
    
    if vectorstore_action == "Create Vectorstore":
        selected_llm_model = st.selectbox("🔹 Select LLM Model", llm_models, index=llm_models.index(st.session_state.llm_model))
        selected_embeddings_model = st.selectbox("🔹 Select Embeddings Model", embeddings_models, index=embeddings_models.index(st.session_state.embeddings_model))
        selected_parsing_method = st.selectbox("🔹 Select Parsing Method", parsing_methods, index=parsing_methods.index(st.session_state.parsing_method))
        uploaded_files = st.file_uploader("📂 Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    else:
        recent_llm = list(config.get("llm_model", {}).keys())[0]
        recent_embeddings = config.get("embeddings_model", {}).get(st.session_state.embeddings_model, "")
        
        st.write(f"🔹 **LLM Model:** {recent_llm}")
        st.write(f"🔹 **Embeddings Model:** {recent_embeddings}")
        uploaded_files = None

    if vectorstore_action == "Create Vectorstore" and st.button("⚡ Create Vectorstore"):
        config["llm_model"] = {selected_llm_model: available_models["llm_models"][selected_llm_model]}
        config["embeddings_model"] = {selected_embeddings_model: available_models["embeddings_models"][selected_embeddings_model]}
        config["parsing_method"] = selected_parsing_method
        save_json(CONFIG_FILE, config)
        
        if uploaded_files:
            st.write("📄 Processing uploaded files...")
            temp_dir = tempfile.gettempdir()
            file_paths = [os.path.join(temp_dir, uploaded_file.name) for uploaded_file in uploaded_files]
            for uploaded_file, file_path in zip(uploaded_files, file_paths):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            try:
                st.session_state.vectorstore = vectorstore_pipeline(
                    embeddings_model_name=available_models["embeddings_models"][selected_embeddings_model],
                    llm_model_name=available_models["llm_models"][selected_llm_model]["huggingface"],
                    file_paths=file_paths,
                    enrich_method="keywords",
                    parsing_method=selected_parsing_method,
                    use_gpu=False
                )
                st.session_state.vectorstore_created = True
                st.success("✅ Vectorstore created successfully!")
            except Exception as e:
                st.error(f"❌ Error creating vectorstore: {e}")
        else:
            st.warning("⚠️ Please upload PDFs before creating a vectorstore.")

    if vectorstore_action == "Load Existing Vectorstore" and not st.session_state.vectorstore_created and st.button("⚡ Load Vectorstore"):
        try:
            st.session_state.vectorstore = load_vectorstore(
                embeddings_model_name=recent_embeddings,
                store_path="faiss_index",
                use_gpu=False
            )
            st.session_state.vectorstore_created = True
            st.success("✅ Loaded existing vectorstore successfully!")
        except FileNotFoundError:
            st.error("❌ Vectorstore not found. Please create a vectorstore first.")
        except Exception as e:
            st.error(f"❌ Failed to load vectorstore: {e}")

    # Initializing Graph
    if st.session_state.vectorstore_created:
        from rag import initialize_graph
        graph, config = initialize_graph(st.session_state.vectorstore, st.session_state.memory)

# Chat UI
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("⏳ Thinking..."):
                inputs = {"messages": [{"role": "user", "content": prompt}], "max_retries": 3}
                response = graph.invoke(inputs, config=config)
                bot_reply = response["messages"][-1].content
                st.write(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})