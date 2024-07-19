import streamlit as st

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
    st.session_state.uploaded_image = None

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "image":
                st.image(message["content"])
            else:
                st.write(message["content"])
