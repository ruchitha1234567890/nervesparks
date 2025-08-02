# Visual Document Analysis RAG System

A RAG-based demo that processes PDFs and images to extract information from tables, charts, and text content.

## Features
- Multi-format document processing
- OCR for scanned documents
- Embedding and retrieval using ChromaDB
- Question-answering with OpenAI GPT

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuration
Set your OpenAI API key in `rag_pipeline.py` or use environment variables.