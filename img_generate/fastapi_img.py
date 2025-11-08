import os
import requests
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models for Request and Response ---

class PromptRequest(BaseModel):
    """Defines the structure for the incoming request body."""
    prompt: str

class ImageResponse(BaseModel):
    """Defines the structure for the outgoing response, sending the image as a Base64 string."""
    image_base64: str

# --- FastAPI App Initialization ---

app = FastAPI(
    title="AI Image Generator API",
    description="An API that generates images from text prompts using Hugging Face.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---

# List of allowed origins (your frontend's URL)
origins = [
    "http://localhost",
    "http://localhost:3000", # Default for Create React App
    "http://localhost:5173", # Default for Vite
    # Add your deployed frontend URL here for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- API Endpoint ---

@app.post("/api/v1/generate-image", response_model=ImageResponse, tags=["Image Generation"])
async def generate_image(request: PromptRequest):
    """
    Takes a text prompt and returns a Base64 encoded image.
    This corresponds to the "image-generator" tool ID.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY is not configured on the server.")

    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": request.prompt}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        encoded_image = base64.b64encode(response.content).decode("utf-8")
        
        return ImageResponse(image_base64=encoded_image)

    except requests.exceptions.HTTPError as e:
        # Handle specific API errors from Hugging Face
        if e.response.status_code == 503:
            raise HTTPException(status_code=503, detail="The model is currently loading on Hugging Face. Please try again in a moment.")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from Hugging Face API: {e.response.text}")
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Health check endpoint
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running"}
