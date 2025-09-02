import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="openrouter/gpt-4o-mini",
    temperature=0.5,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
)



