Document Question Answering API

A FastAPI-based Question Answering System that supports document ingestion, vector search, and LLM-powered answering using OpenAI embeddings and Pinecone vector database.

Overview

This project implements a Document-Based Question Answering API that allows you to:

Upload and index text documents.

Perform semantic vector search using embeddings.

Generate natural language answers using an LLM (OpenAI GPT models).

Manage stored documents (list and delete).

It is designed according to the assessment requirements for:

Building a document-based question answering API combining vector search and LLMs.


-Tech Stack
Component	Technology Used
Language	Python 3.10+
Framework	FastAPI
Vector Database	Pinecone
Embeddings	OpenAI text-embedding-ada-002
LLM	OpenAI GPT (via agent.py)
Environment Management	.env file
Async Execution	Supported with FastAPI and asyncio
Containerization
Docker + Docker Compos



Setup Instructions

1. Clone the Repository
    git clone https://github.com/<your-username>/document-qa-api.git
    cd document-qa-api

2. Create and Activate Virtual Environment
    python3 -m venv venv
    source venv/bin/activate

3. Install Dependencies
    pip install -r requirements.txt

4. Create a .env file in the app/ directory:

    OPENAI_API_KEY="your-openai-api-key"
    PINECONE_API_KEY="your-pinecone-api-key"
    PINECONE_INDEX="assessment-index"
    PINECONE_ENV="us-east-1"
    DEBUG=True

5. Running the Application
    cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

OPTION 2
RUN WITH DOCKER COMPOSE 


    Prerequisites

    Make sure you have installed:

    Docker

    Docker Compose

    Check versions:

    docker -v
    docker compose version

    commands :
    docker-compose build --> this will create the build of complete project and install all dependancies 
    docker-compose up  --> this will run that container 

6. Visit:
    Home Page: http://localhost:8000/

    Chat Page: http://localhost:8000/chat

7. Upload Documents

    POST /documents

    Uploads one or more files and indexes them.

    Request Example (multipart/form-data):

    files: [file1.txt, file2.pdf]


    Response:

    {
    "total_documents": 2,
    "results": [
        {
        "doc_id": "5bde4f1b-13a8-4e5f-8fa4-ccbd787e1c21",
        "title": "file1.txt",
        "chunks": 8,
        "message": "Document embedded successfully"
        }
    ]
    }


8. List Documents

    GET /documents

    Returns all stored document IDs, titles, and chunk counts.

    Response Example:

    {
    "total_documents": 2,
    "documents": [
        {
        "doc_id": "abc123",
        "title": "climate.txt",
        "chunks": 5
        }
    ]
    }

9. Delete Document

    DELETE /documents/{doc_id}

    Deletes the document and all its embeddings.

    Response:

    {
    "message": "Document with ID abc123 deleted successfully.",
    "response": {}
    }

10. Chat via WebSocket

    Endpoint: /ws/chat

    You can connect to the chat using a WebSocket client or frontend (chat.html).

    Sample Flow:

    # Client → Server
    {"input": "What is the main topic of climate.txt?"}

    # Server → Client
    {"type": "answer", "message": "The document discusses climate change and its global impact."}


    Errors are streamed as:

    {"type": "error", "message": "Error while processing query"}



SELECTION OF TECH 

I have followed all the requirements outlined in the assessment and made a few enhancements to improve performance and scalability.

WebSocket for Real-Time Streaming
Instead of using a traditional POST API for search, I implemented WebSockets. This makes the system more efficient and enables real-time communication between the backend and frontend. I also customized the LangChain streaming handler to support real-time token-level streaming, ensuring smooth integration with the frontend interface.

Pinecone Vector Database (vs. Local FAISS)
For vector storage, I used Pinecone instead of a local FAISS database. Pinecone offers faster query performance, better scalability, and eliminates the need for local index management. It’s also free for small-scale tasks, making it a practical and efficient choice.

Custom OpenAI Agent for Semantic Search
Rather than relying on the OpenAI Assistant API, I built a custom OpenAI agent for semantic search. This approach improves speed and flexibility, allowing for advanced vector-based semantic retrieval and memory management. The agent’s architecture follows a production-grade design that can be extended easily for future enhancements.
