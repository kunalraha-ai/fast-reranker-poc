import os

class Config:
    """
    Central configuration for the Pax Historia Reranker.
    """
    
    # Model Settings
    # We use BGE-M3 for the best balance of speed vs accuracy
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"
    
    # Inference Settings
    # Keep max_length reasonable to avoid OOM (Out of Memory) errors
    MAX_LENGTH = 512
    BATCH_SIZE = 4 
    
    # Device Management
    # Auto-detects if GPU is available, otherwise defaults to CPU
    DEVICE = "cuda" if os.environ.get("USE_GPU") == "true" else "cpu"

    # API Settings
    API_PORT = 8000
    API_HOST = "0.0.0.0"
