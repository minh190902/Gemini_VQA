import os
import json
import streamlit as st

def generate_gemini_response():

    string_dialogue = """System: You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.
                        Some rules to follow:
                            1. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
                            2. You must focus on OCR task.
                            3. Answer question from <User> with clarification and high precision, using the provided <Image>.
                            4. Answer in vietnamese language.
                            5. Have full of sentence answer: S + V + adj + ...
                            """

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            if dict_message.get("type") == "image":
                string_dialogue += "User: <Image>\n\n"
            else:
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n" 
        
    return string_dialogue
