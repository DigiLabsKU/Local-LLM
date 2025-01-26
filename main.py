import streamlit as st
from rag import create_agent, clear_mem

# Initialize our conversation chain and refresh dropbox access token
agent = create_agent()

def clear_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    clear_mem()

if __name__ == '__main__':

    st.set_page_config(page_title='💬 Local RAG')

    with st.sidebar:
        st.title('💬 Local RAG')
        st.markdown("""
        The following chatbot is a simple RAG application allowing users to chat with their documents completely offline and private. 

        **How to Use:** Type your question in the chatbox, and the chatbot will try to answer your questions using the provided documents. If you wish to start a new conversation, use the 'Clear Conversation' button below.

        **Disclaimer:** While the chatbot strives for accuracy, it might not always be correct. Always refer to official sources for critical information.
        """)
        
        # Clear conversation button
        st.button('Clear Conversation', on_click=clear_conversation)

        # Add a link to the questionnaire (feedback survey)
        st.markdown("📋 [Fill out our feedback survey](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAANAAdAMTRFUQk1JQTFFRE1DTVlBNVpaSTMzRlAzQzJTWS4u)")

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
                response = agent.run(input=prompt)
                if response.endswith("'''"):
                    response = response[:-3]
                st.write(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
