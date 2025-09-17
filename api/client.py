import requests
import streamlit as st


def make_api_request_and_parse(endpoint_url, input_text, field_name = "topic"):
    """
    Sends a POST request to a specified endpoint and safely parses the JSON response.
    """
    try:
        payload = {"input": {field_name: input_text}}

        response = requests.post(endpoint_url, json=payload)
        
        response.raise_for_status()
        
        data = response.json()
        
        # Safely access the nested output content
        output = data.get("output", {})
        if isinstance(output, dict):
            return output.get("content", "Error: 'content' key not found in the response output.")
        
        return output

    except requests.exceptions.RequestException as e:
        return f"API Request Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# --- Specific functions for each endpoint ---

def get_essay_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/essay/invoke", input_text, field_name="topic")

def get_poem_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/poem/invoke", input_text, field_name="topic")

def get_chat_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/chat/invoke", input_text, field_name="question")

def get_expert_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/expert/invoke", input_text, field_name = "question")


# --- Streamlit UI Part ---
st.title("Langchain Demo with API Server")

input_text_essay = st.text_input('Write an essay on:')
if input_text_essay:
    st.write(get_essay_response(input_text_essay))

input_text_poem = st.text_input('Write a poem on:')
if input_text_poem:
    st.write(get_poem_response(input_text_poem))

input_text_chat = st.text_input('Talk with your virtual friend:')
if input_text_chat:
    st.write(get_chat_response(input_text_chat))

input_text_expert = st.text_input('What you wanna know?')
if input_text_expert:
    st.write(get_expert_response(input_text_expert))