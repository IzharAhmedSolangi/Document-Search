from PyPDF2 import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import os, json, pinecone
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from config import client


def extract_text(file: UploadFile):
    if file.filename.endswith(".pdf"):
        return "".join([p.extract_text() or "" for p in PdfReader(file.file).pages])
    elif file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    elif file.filename.endswith(".json"):
        return json.load(file.file).get("content", "")
    raise HTTPException(400, "Unsupported file type (use .pdf, .txt, or .json)")


def embed(text: str):
    return client.embeddings.create(input=text, model="text-embedding-ada-002").data[0].embedding
