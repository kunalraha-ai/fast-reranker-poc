# 🔍 Semantic Reranker Microservice

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)

> **"Search is not just about matching keywords; it is about understanding intent."**

## 🚀 Overview
This project is a **Semantic Reranking Engine** designed to improve the relevance of Information Retrieval (IR) systems. It takes a user query and a list of potential documents, then uses a Deep Learning model (Cross-Encoder) to re-order the documents based on semantic similarity.

This mimics the architecture that is used in modern **RAG (Retrieval-Augmented Generation)** pipelines to ensure LLMs receive only the most relevant context.

## 🏗️ Architecture
The application follows a microservices pattern, fully containerized with Docker.

```mermaid
graph LR
    A["User (Streamlit UI)"] -->|Query + Docs| B("FastAPI Backend")
    B -->|Inference| C{"Transformer Model"}
    C -->|Relevance Scores| B
    B -->|Ranked Results| A
```
