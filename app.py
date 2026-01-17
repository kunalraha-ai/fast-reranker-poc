from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from typing import List
import uvicorn
import time

# --- CONFIGURATION ---
# We use TinyBERT because it is 10x faster than standard BERT.
# Perfect for Real-Time Reranking.
MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2'

# 1. Initialize the App
app = FastAPI(
    title="Pax Historia Reranker API",
    description="Optimized Reranking Service to cut Latency & Costs.",
    version="1.0.0"
)

# 2. Load the Model ONCE (Global Load)
# This prevents reloading the model for every user (which causes crashes).
print(f"⚡ Loading AI Model: {MODEL_NAME}...")
model = CrossEncoder(MODEL_NAME, max_length=512)
print("✅ Model Loaded. Server is ready.")

# 3. Define the Input Data Structure
class RerankRequest(BaseModel):
    query: str              # The user's question
    documents: List[str]    # The raw retrieved documents (context)
    top_k: int = 3          # How many top results to keep

# 4. The Reranking Endpoint
@app.post("/rerank")
async def rerank(payload: RerankRequest):
    try:
        start_time = time.time()
        
        # Validating Input
        if not payload.documents:
            return {"top_results": [], "message": "No documents provided."}

        # Prepare pairs for the model: [ [Query, Doc1], [Query, Doc2]... ]
        pairs = [[payload.query, doc] for doc in payload.documents]
        
        # Predict scores (The AI Magic)
        scores = model.predict(pairs)
        
        # Organize results
        results = []
        for i, score in enumerate(scores):
            results.append({
                "text": payload.documents[i],
                "score": float(score),
                "original_index": i
            })
            
        # Sort by score (Highest confidence first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Slice top K
        top_results = results[:payload.top_k]
        
        # Calculate Latency
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "top_results": top_results,
            "latency_ms": round(process_time * 1000, 2)  # In Milliseconds
        }

    except Exception as e:
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. Simple Health Check
@app.get("/")
def home():
    return {"status": "online", "message": "Reranker API is live. 🟢"}