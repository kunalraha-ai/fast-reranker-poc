# Fast Reranker POC (BGE-M3)

## Objective
To build a high-precision, low-latency reranking layer for RAG pipelines, specifically optimizing for:
- **Latency:** < 50ms per query.
- **Cost:** Removing dependency on GPT-4 for filtering.
- **Accuracy:** Leveraging Cross-Encoders for final candidate selection.

## Tech Stack
- **Model:** BGE-M3 (BAAI/bge-m3)
- **Framework:** PyTorch / ONNX Runtime (for quantization)
- **Vector DB:** Pinecone (simulated)

## Roadmap
- [ ] Setup Environment & Dependencies
- [ ] Baseline Benchmark (HuggingFace Transformers)
- [ ] Optimization Layer (ONNX/Quantization)
- [ ] API Wrapper (FastAPI)
- [ ] Dockerization for Deployment
