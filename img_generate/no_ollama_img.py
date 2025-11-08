import streamlit as st
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
#import google.generativeai as genai
import requests

# Load environment variables from a .env file
load_dotenv()

# --- Configure Google Generative AI ---
# Configure the API key from environment variables
# try:
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# except TypeError:
#     st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
#     st.stop()

# --- Google AI (Imagen 3) for Image Generation ---
# --- Hugging Face API for Image Generation ---
def get_image_from_api(prompt: str):
    """
    Calls the Hugging Face Inference API to generate an image.
    """
    st.write("☁️ Sending request to Hugging Face...")
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
        "inputs": prompt
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
st.set_page_config(page_title="AI Image Generator", layout="wide")
st.title("🎨 AI Image Generator")
st.markdown("Enter a description below to generate an image")

user_prompt = st.text_input("Describe the image you want to create:", placeholder="e.g., 'A majestic lion wearing a crown, cinematic photo'")

if st.button("Generate Image", type="primary"):
    if user_prompt:
        with st.spinner("Creating your image..."):
            image_bytes = get_image_from_api(user_prompt)
            
            if image_bytes:
                try:
                    image = Image.open(image_bytes)
                    
                    st.markdown("### 🖼️ Generated Image")
                    st.image(image, caption=user_prompt, use_container_width=True)
                    st.success("Image generated successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred while displaying the image: {e}")
    else:
        st.warning("Please enter a description to generate an image.")