import streamlit as st
from rag import handle_user_query_with_graph
from vectorstore import vectorstore_pipeline, load_existing_vectorstore
from langchain_community.vectorstores import FAISS
import os 
import tempfile

def clear_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


if __name__ == '__main__':

    st.set_page_config(page_title='💬 Local RAG')

    with st.sidebar:
        st.title('💬 Local RAG')
        st.markdown("""
        The following chatbot is a simple RAG application allowing users to chat with their documents completely offline and private. 

        **How to Use:** 
                    
        Upload your documents here by either dragging them into the blue box below or using the file system. 
        Once the documents have been parsed and loaded, type your question in the chatbox, and the chatbot will try to answer your questions using the provided documents. If you wish to start a new conversation, use the 'Clear Conversation' button below.

        **Disclaimer:** While the chatbot strives for accuracy, it might not always be correct. Always refer to official sources for critical information.
        """)

        # File uploading logic
        uploaded_files = st.file_uploader("Upload your documents (PDFs)", type=["pdf"], accept_multiple_files=True)

        # Vectorstore management
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None

        if 'vectorstore_created' not in st.session_state:
            st.session_state.vectorstore_created = False  # Flag to track if vectorstore is created

        if uploaded_files and not st.session_state.vectorstore_created:
            # Create a temporary directory
            temp_dir = tempfile.gettempdir()

            file_paths = [os.path.join(temp_dir, uploaded_file.name) for uploaded_file in uploaded_files]
            for uploaded_file, file_path in zip(uploaded_files, file_paths):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Create a vectorstore using the uploaded files
            st.write("Processing uploaded files...")
            embeddings_model_name = "all-MiniLM-L6-v2"  # Use the same embeddings model name as defined earlier
            store_path = "faiss_index"

            try:
                st.session_state.vectorstore = vectorstore_pipeline(
                    embeddings_model_name=embeddings_model_name,
                    llm_model="meta-llama/Llama-3.2-3B",
                    file_paths=file_paths,
                    enrich_method="keywords",
                    store_path=store_path,
                    use_gpu=False
                )
                st.session_state.vectorstore_created = True  # Mark the flag as True
                st.write("Vectorstore created successfully!")
            except Exception as e:
                st.error(f"Error creating vectorstore: {e}")  

        # Optionally load an existing vectorstore
        load_existing = st.checkbox("Load existing vectorstore")
        if load_existing and not st.session_state.vectorstore_created:
            store_path = "faiss_index"
            try:
                st.session_state.vectorstore = load_existing_vectorstore(
                    embeddings_model_name="all-MiniLM-L6-v2",
                    store_path=store_path,
                    use_gpu=False
                )
                st.session_state.vectorstore_created = True  # Mark the flag as True
                st.write("Loaded existing vectorstore successfully!")
            except Exception as e:
                st.error(f"Failed to load vectorstore: {e}")

        # Clear conversation button
        st.button('Clear Conversation', on_click=clear_conversation)

    # Initialize conversation state if not already done
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


    # Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Get user input
    if prompt := st.chat_input('Your message...'):
        # Display user input
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner('Please Wait...'):
                try:
                    response = handle_user_query_with_graph(prompt)
                    if response.endswith("'''"):
                        response = response[:-3]
                    st.write(response)
                    st.session_state.messages.append({'role': 'assistant', 'content': response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.messages.append({'role': 'assistant', 'content': 'Sorry, I encountered an error.'})
        else:
            st.warning("Please upload files and/or load a vectorstore to begin.")
