import requests
import streamlit as st

# A single, robust function to handle all API requests and parse the response
def make_api_request_and_parse(endpoint_url, topic):
    """
    Sends a POST request to a specified endpoint and safely parses the JSON response.
    """
    try:
        # The JSON payload structure for the API
        payload = {"input": {'topic': topic}}
        
        # Make the POST request
        response = requests.post(endpoint_url, json=payload)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        data = response.json()
        
        # Safely access the nested output content
        output = data.get("output", {})
        if isinstance(output, dict):
            return output.get("content", "Error: 'content' key not found in the response output.")
        
        # If output is not a dictionary (e.g., just a string), return it directly
        return output

    except requests.exceptions.RequestException as e:
        return f"API Request Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# --- Specific functions for each endpoint ---

def get_essay_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/essay/invoke", input_text)

def get_poem_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/poem/invoke", input_text)

def get_chat_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/chat/invoke", input_text)

def get_expert_response(input_text):
    return make_api_request_and_parse("http://localhost:8000/expert/invoke", input_text)


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