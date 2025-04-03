import streamlit as st
import os
import tempfile
from vectorstore import vectorstore_pipeline, CustomMultiVectorStore
from model_configuration import load_json, save_json
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import tempfile

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

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Sidebar UI
with st.sidebar:
    st.title("💬 Local RAG")
    st.markdown("""Your document-based RAG chatbot. Select models, upload PDFs, and start chatting.""")

    if st.button("🗑️ Clear Conversation History"):
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.session_state.memory = MemorySaver() # Reintialize memory
        st.success("🗑️ Conversation history cleared!")

    vectorstore_action = st.radio("Choose an action", ("Create Vectorstore", "Load Existing Vectorstore"))
    
    if vectorstore_action == "Create Vectorstore":
        selected_llm_model = st.selectbox("🔹 Select LLM Model", llm_models, index=llm_models.index(st.session_state.llm_model))
        selected_embeddings_model = st.selectbox("🔹 Select Embeddings Model", embeddings_models, index=embeddings_models.index(st.session_state.embeddings_model))
        selected_parsing_method = st.selectbox("🔹 Select Parsing Method", parsing_methods, index=parsing_methods.index(st.session_state.parsing_method))
        uploaded_files = st.file_uploader("📂 Upload PDFs", type=["pdf", "txt", "pptx", "docx", "HTML", "xls"], accept_multiple_files=True)
    
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
        if "gpt" not in selected_llm_model:
            selected_llm_model = available_models["llm_models"][selected_llm_model]["huggingface"]
        save_json(CONFIG_FILE, config)
        
        if uploaded_files:
            st.write("📄 Processing uploaded files...")
            temp_dir = tempfile.gettempdir()
            file_paths = [os.path.join(temp_dir, uploaded_file.name) for uploaded_file in uploaded_files]
            for uploaded_file, file_path in zip(uploaded_files, file_paths):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.session_state.multi_vector_store = vectorstore_pipeline(
                embeddings_model_name=available_models["embeddings_models"][selected_embeddings_model],
                llm_model_name=selected_llm_model,
                file_paths=file_paths,
                parsing_method=selected_parsing_method,
                use_gpu=False
            )
            st.session_state.vectorstore_created = True
            st.success(f"✅ {st.session_state.multi_vector_store.num_vectorstores()} Vectorstore(s) created successfully!")

        else:
            st.warning("⚠️ Please upload PDFs before creating a vectorstore.")

    if vectorstore_action == "Load Existing Vectorstore":
        load_btn = st.button("⚡ Load Vectorstore", disabled=st.session_state.vectorstore_created)

        if not st.session_state.vectorstore_created and load_btn:
            try:
                languages = config["languages"]
                st.session_state.multi_vector_store = CustomMultiVectorStore(
                    embeddings_model_name=recent_embeddings,
                    languages=languages
                )
                st.session_state.multi_vector_store.load_vectorstores(languages)
                st.session_state.vectorstore_created = True
                st.success(f"✅ Loaded existing {st.session_state.multi_vector_store.num_vectorstores()} vectorstore(s) successfully!")
            except FileNotFoundError:
                st.error("❌ Vectorstore not found. Please create a vectorstore first.")
            except Exception as e:
                st.error(f"❌ Failed to load vectorstore: {e}")

        # If already loaded, offer to extend it
        if st.session_state.vectorstore_created:
            st.markdown("### ➕ Extend Loaded Vector Store")
            extension_files = st.file_uploader("📂 Upload Files to Extend Vectorstore", type=["pdf", "txt", "pptx", "docx", "HTML", "xls"], accept_multiple_files=True, key="extend_uploader")
            if st.button("📌 Extend Vector Store"):
                if extension_files:
                    temp_dir = tempfile.gettempdir()
                    file_paths = [os.path.join(temp_dir, uploaded_file.name) for uploaded_file in extension_files]
                    for uploaded_file, file_path in zip(extension_files, file_paths):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    from vectorstore import extend_multi_vector_store

                    st.session_state.multi_vector_store = extend_multi_vector_store(
                        st.session_state.multi_vector_store,
                        llm_model_name=recent_llm,
                        file_paths=file_paths,
                        parsing_method=config.get("parsing_method", "local")
                    )

                    from rag import initialize_graph
                    graph, config = initialize_graph(st.session_state.multi_vector_store, st.session_state.memory)
                    st.success("✅ Vectorstore extended successfully!")
                else:
                    st.warning("⚠️ Please upload files to extend the vectorstore.")

    # Initializing Graph
    if st.session_state.vectorstore_created:
        from rag import initialize_graph
        graph, config = initialize_graph(st.session_state.multi_vector_store, st.session_state.memory)

# Chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"), st.empty():
        with st.spinner("⏳ Thinking..."):
            inputs = {"messages": [{"role": "user", "content": prompt}], "max_retries": 3}
            response = graph.invoke(inputs, config=config)
            bot_reply = response["messages"][-1].content
        st.markdown(bot_reply, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# Prepare conversation export content
conversation_text = "\n------------------------\n".join(
        f'User: {st.session_state.messages[i]["content"]}\n\nAssistant: {st.session_state.messages[i+1]["content"]}'
        for i in range(1, len(st.session_state.messages)-1, 2)
    )

if len(st.session_state.messages) > 1:
    with st.container():
        st.markdown(" ")
        st.download_button(
            label="📥 Export Conversation",
            data=conversation_text,
            file_name=f'conversation_{datetime.now().strftime("%H_%M_%S_%d_%m_%Y")}.txt',
            mime="text/plain"
        )
