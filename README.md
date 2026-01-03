# Intelligent Complaint Analysis – Task 1 & 2

This repository contains the implementation of **Task 1 (EDA & Preprocessing)** and **Task 2 (Text Chunking, Embedding, and Vector Store Indexing)** for CFPB complaint data. The goal is to prepare complaint narratives for semantic search and downstream RAG systems.

---

## Tasks Overview

**Task 1 – EDA & Preprocessing:**  
- Load the raw CFPB dataset (`data/raw/complaints.csv`)  
- Explore complaint distribution across five products: Credit Cards, Personal Loans, Savings Accounts, Money Transfers, Checking Accounts  
- Analyze narrative lengths and filter out missing or empty narratives  
- Clean text by lowercasing and removing special characters/boilerplate  
- Save cleaned data to `data/processed/filtered_complaints.csv`  

**Task 2 – Chunking & Embedding:**  
- Create a stratified sample of ~10,000–15,000 complaints (`data/processed/sample_complaints.csv`)  
- Split long narratives into overlapping chunks (500 characters, 50 overlap) and save to `data/processed/chunks.csv`  
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`  
- Store embeddings with metadata in a persistent ChromaDB vector store (`vector_store/`)  



