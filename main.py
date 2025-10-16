from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from utills import extract_text, embed
from config import index
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from callback_handlers import FastAPIStreamingCallbackHandler
from agent import create_agent_executor
from collections import defaultdict
from typing import List
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Document Retriever APP")

templates = Jinja2Templates(directory="templates")
origins = [
    "http://127.0.0.1:8000",   
    "http://0.0.0.0:8000",   
    "http://localhost:8000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,
    allow_methods=["*"],              
    allow_headers=["*"],            
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("documents.html", {"request": request, "title": "Home Page"})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "title": "Chat Page"})



@app.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    print(f"Received {len(files)} file(s): {[f.filename for f in files]}")
    try:
        all_results = []

        for file in files:
            doc_id = str(uuid.uuid4())
            text = extract_text(file)
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

            vectors = []
            for i, chunk in enumerate(chunks):
                vectors.append((
                    f"{doc_id}_{i}",
                    embed(chunk),
                    {
                        "doc_id": doc_id,
                        "title": file.filename,
                        "chunk_index": i,
                        "chunk": chunk[:200],
                        "source": "user_upload"
                    }
                ))

            index.upsert(vectors=vectors)
            all_results.append({
                "doc_id": doc_id,
                "title": file.filename,
                "chunks": len(chunks),
                "message": "Document embedded successfully"
            })

        return {
            "total_documents": len(all_results),
            "results": all_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

@app.get("/documents")
async def list_documents():
    try:
      
        response = index.list() 
        
        docs_map = defaultdict(lambda: {"title": None, "chunks": 0})
        cursor = None

        while True:
            results = index.query(
                vector=[0.0] * 1536, 
                top_k=10000,
                include_metadata=True,
                filter={},
                namespace="", 
                **({"cursor": cursor} if cursor else {})
            )

            for match in results["matches"]:
                metadata = match["metadata"]
                doc_id = metadata.get("doc_id")
                title = metadata.get("title")
                docs_map[doc_id]["title"] = title
                docs_map[doc_id]["chunks"] += 1

            cursor = results.get("next_cursor")
            if not cursor:
                break

        documents = [
            {
                "doc_id": doc_id,
                "title": info["title"],
                "chunks": info["chunks"]
            }
            for doc_id, info in docs_map.items()
        ]

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        
        delete_response = index.delete(
            filter={"doc_id": {"$eq": doc_id}} 
        )

        return {
            "message": f"Document with ID {doc_id} deleted successfully.",
            "response": delete_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("input", "").strip()

            if not message:
                await websocket.send_json({"type": "error", "message": "Empty input received."})
                continue
            handler = FastAPIStreamingCallbackHandler(websocket)
            agent_executor = create_agent_executor(callbacks=[handler])

            try:
                response = await agent_executor.ainvoke({"input": message})
                answer = response["output"]

                await websocket.send_json({
                    "type": "answer",
                    "message": answer
                })

            except Exception as e:
                error_msg = f"Error while processing query: {str(e)}"
                print(f"{error_msg}")
                await websocket.send_json({"type": "error", "message": error_msg})

    except WebSocketDisconnect:
        print(" Client disconnected")
        