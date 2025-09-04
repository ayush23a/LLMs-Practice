import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import os
from langchain_ollama import OllamaLLM as Ollama 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Import and Configure Google Generative AI ---
import google.generativeai as genai

# Configure the API key from environment variables
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except TypeError:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()


# --- Ollama and Langchain for Prompt Refinement ---
# Initialize the local LLM
llm = Ollama(model="gemma3") 

def get_refined_prompt(user_prompt: str) -> str:
    """
    Refines the user's prompt into a detailed description for image generation.
    """
    st.write(" Refining prompt with local Gemma model...")
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert image description writer. Modify the user prompt into a highly detailed, one-paragraph description for an AI image generator. The description should be vivid and include details about the artistic style, lighting, mood, and composition."),
        ("human", "User prompt: {user_prompt}"),
    ])
    
    refinement_chain = refinement_prompt | llm
    refined_prompt = refinement_chain.invoke({"user_prompt": user_prompt})
    return refined_prompt

# --- Google AI (Imagen 3) for Image Generation (Updated Function) ---
def get_image_from_api(refined_prompt: str):
    """
    Calls the Stability Diffusion API to generate an image from the refined prompt.
    """
    st.write(" Sending request to Hugging Face")
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        st.error("HUGGINGFACE_API_KEY not found in .env file.")
        return None

    # This is the endpoint for the popular Stable Diffusion XL model
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "inputs": refined_prompt
    }

    try:
        # Make the API call
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an error for bad status codes

        # The response body is the image data directly
        return BytesIO(response.content)

    except requests.exceptions.HTTPError as e:
        # Provide a helpful error for the common "model loading" issue
        if response.status_code == 503:
             st.error("Model is currently loading on Hugging Face. Please wait a moment and try again.")
        else:
            st.error(f"Error generating image from Hugging Face: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Image Generator Agent", layout="wide")
st.title("🤖 AI Image Generator Agent")
st.markdown("Enter a short description below. A local AI will refine it into a detailed prompt, then a powerful cloud AI will generate the image.")

user_prompt = st.text_input("Describe the image you want to create:", placeholder="e.g., 'a cat wearing a wizard hat'")

if st.button("Generate Image", type="primary"):
    if user_prompt:
        with st.spinner("Thinking and creating..."):
            refined_prompt = get_refined_prompt(user_prompt)
            
            st.markdown("### ✨ Refined Prompt")
            st.write(refined_prompt)
            
            image_bytes = get_image_from_api(refined_prompt)
            
            if image_bytes:
                try:
                    image = Image.open(image_bytes)
                    
                    st.markdown("### 🖼️ Generated Image")
                    st.image(image, use_column_width=True)
                    st.success("Image generated successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred while displaying the image: {e}")
    else:
        st.warning("Please enter a description to generate an image.")