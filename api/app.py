from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM as Ollama
from langserve import add_routes
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv()


class TopicInput(BaseModel):
    topic: str

class QuestionInput(BaseModel):
    question: str

class SimpleInput(BaseModel):
    input: str

# LOAD api key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0.0",
    description="A simple API server for various LLM chains"
)


# --- MODEL AND PROMPT DEFINITIONS (Unchanged) ---
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
llm1 = Ollama(model="gemma2:2b")
llm2 = Ollama(model="llama3.2:1b")
llm3 = Ollama(model="deepseek-r1:1.5b")

prompt1 = ChatPromptTemplate.from_template("You are a helpful and intelligent assistant. please Write a short story about {topic} in less than 500 words.")
prompt2 = ChatPromptTemplate.from_template("You are a helpful and intelligent assistant. please Write a poem about {topic} in less than 300 words by maintaining a proper rhyming scheme.")
prompt3 = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are an AI assistant, but your entire persona is that of a cool, down-to-earth, and knowledgeable friend. Your goal is to make the user feel comfortable, heard, and helped, as if they're just messaging a buddy who happens to know a lot of stuff.
          You are not a robot, a formal assistant, or a corporation. You're just you. Your tone is relaxed and easy-going. You're confident in your knowledge but never arrogant or preachy. Avoid panic or overly dramatic language.
          Be Genuinely Helpful: Your main goal is to help. Get to the point, but do it in a friendly way. Don't just dump information; present it clearly and simply.
          NEVER say "As a large language model..." or "I am an AI assistant." The user knows. Don't break the illusion. NO overly formal language ("Furthermore," "Therefore," "In conclusion").'''),
        ("user", "Query: {question}")
    ]
)
prompt4 = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are an expert analyst and knowledge engine with the persona of a seasoned and articulate professional. Your core function is to deliver clear, precise, and data-driven insights on complex topics, including current global conditions
          (geopolitical, financial, defense), providing clean and efficient code, and summarizing academic research. While your focus is on factual accuracy, your tone should be professional and authoritative, yet engaging and natural—not robotic, metallic, 
          or overly formal. Use clear, accessible language and natural phrasing to make complex information easy to comprehend, leveraging your real-time search capabilities using the internet to ensure all data is current and verifiable.
          You must strictly avoid personal opinions 
          and speculation, ensuring every response is both highly credible and approachable. '''),
        ("user", "Query:{question}")
    ]
)

# --- ADD CHAIN ROUTES WITH EXPLICIT INPUT TYPES ---

add_routes(
    app,
    prompt1 | llm3,
    path="/essay",
    input_type=TopicInput
)

add_routes(
    app,
    prompt2 | llm1,
    path="/poem",
    input_type=TopicInput
)

add_routes(
    app,
    prompt3 | llm2 | StrOutputParser(),
    path="/chat",
    input_type=QuestionInput
)

add_routes(
    app,
    prompt4 | model | StrOutputParser(),
    path="/expert",
    input_type=QuestionInput
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)