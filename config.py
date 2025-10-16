from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env") 

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_ENV: str
    DEBUG: bool = False
    
    

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = settings.PINECONE_INDEX
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENV),
    )
index = pc.Index(index_name)