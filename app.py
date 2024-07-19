import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from gemini_response import generate_gemini_response
from chat_history import clear_chat_history, initialize_chat, display_chat_messages
import google.generativeai as ggi
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.schema import ImageDocument
# Import vector database functions
# from vector_db import save_chat_history, load_chat_history, get_embedding, search_faiss

load_dotenv(".env")

fetcheed_api_key = os.getenv("API_KEY")
ggi.configure(api_key=fetcheed_api_key)

# model = ggi.GenerativeModel("gemini-pro") 
vision_model = ggi.GenerativeModel('gemini-1.5-flash')
chat = vision_model.start_chat()

def LLM_Response(question, img):
    if img is None:
        return chat.send_message(question, stream=True)
    else:
        return chat.send_message([question, img], stream=True)

# Set up Streamlit app
st.set_page_config(page_title="🦙💬 Receipts Chatbot 📖")

# Sidebar configuration
with st.sidebar:
    st.title('🦙💬 Receipts Chatbot 📖')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=1.0, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    st.button('Clear Chat History', on_click=clear_chat_history)

# Initialize chat
initialize_chat()

# Display chat messages
display_chat_messages()

# Main chat application
st.title("Image Question Answering")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Check if an image has been uploaded previously in the session
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# If a new file is uploaded, store it in the session state
if uploaded_file:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption='Uploaded Image', use_column_width=True)
else:
    st.session_state.uploaded_image = None

# Text input for chat
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if prompt:
    st.session_state.messages.append({"role": "user", "content": st.session_state.uploaded_image, "type": "image"})
    st.session_state.messages.append({"role": "user", "content": prompt})


# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        instruct = generate_gemini_response()
        instruct += prompt
        result = LLM_Response(instruct, st.session_state.uploaded_image)
        st.subheader("Response:")
        full_response = " "
        placeholder = st.empty()
        for word in result:
            full_response += word.text
        placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)



