import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import os
import base64
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Ollama and Langchain for Prompt Refinement ---
# Initialize the local LLM
llm = OllamaLLM(model="gemma3") 

def get_refined_prompt(user_prompt: str) -> str:
    """
    Refines the user's prompt into a detailed description for image generation.
    """
    st.write("🖌️ Refining prompt with local Gemma model...")
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert image description writer. Modify the user prompt into a highly detailed, one-paragraph description for an AI image generator. The description should be vivid and include details about the artistic style, lighting, mood, and composition."),
        ("human", "User prompt: {user_prompt}"),
    ])
    
    refinement_chain = refinement_prompt | llm
    refined_prompt = refinement_chain.invoke({"user_prompt": user_prompt})
    return refined_prompt

# --- Stability AI for Image Generation (Updated Function) ---
def get_image_from_api(refined_prompt: str):
    """
    Calls the Stability AI API to generate an image from the refined prompt.
    """
    st.write("☁️ Sending request to Stability AI...")
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        st.error("STABILITY_API_KEY environment variable not set. Please set your API key in the .env file.")
        return None

    engine_id = "sd3-medium"
    url = f"https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "text_prompts": [{"text": refined_prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        data = response.json()
        
        # Get the base64 image data
        image_data = data["artifacts"][0]["base64"]
        
        # Decode the base64 string into bytes and return as a BytesIO stream
        return BytesIO(base64.b64decode(image_data))
        
    except requests.exceptions.HTTPError as e:
        st.error(f"Error generating image from Stability AI: {e.response.text}")
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
            # Error messages are now handled inside the get_image_from_api function
    else:
        st.warning("Please enter a description to generate an image.")