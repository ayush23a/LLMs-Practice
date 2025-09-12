from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn


load_dotenv()

class SimpleInput(BaseModel):
    input : str


# LOAD api key 
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title = "Langchain Server", 
    version = "1.0.0",
    description= "A simple API server"
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest"),
    path = "/gemini",
    input_type = SimpleInput
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
#ollama model
llm = Ollama(model= "gemma3:latest")


prompt1 = ChatPromptTemplate.from_template("You are a helpful and intelligent assistant. please Write a short story about {topic} in less than 500 words.")
prompt2 = ChatPromptTemplate.from_template("You are a helpful and intelligent assistant. please Write a poem about {topic} in less than 300 words by maintaining a proper rhyming scheme.")


add_routes(
    app,
    prompt1|model,
    path = "/essay"
)

add_routes(
    app,
    prompt2|llm,
    path = "/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host = "localhost", port= 8000)


