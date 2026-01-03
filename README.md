
# Intelligent Complaint Analysis – Task 1 & Task 2

This repository contains the implementation of the first phase of the Intelligent Complaint Analysis project for CrediTrust Financial. The goal of this phase is to **clean, preprocess, and transform raw complaint narratives into structured, searchable data** for semantic retrieval.

---

## Project Structure

```

project_root/
├─ data/
│  ├─ raw/                # Original complaint CSV files
│  └─ processed/          # Filtered, cleaned, and chunked datasets
├─ src/
│  ├─ preprocessing.py    # Data loading, filtering, and text cleaning
│  ├─ chunking.py         # Text chunking into overlapping segments
│  └─ vector_store.py     # Embedding and ChromaDB storage
├─ notebooks/
│  └─ Task1_Task2_analysis.ipynb  # Orchestration of preprocessing, chunking, and embedding
├─ vector_store/          # Persistent ChromaDB vector store
└─ README.md

````

---

## Description of Tasks

**Task 1 – Data Preprocessing and EDA**  
- Load the full CFPB complaint dataset.  
- Perform exploratory data analysis (EDA) to understand product distribution and narrative lengths.  
- Filter to the target products (Credit Cards, Personal Loans, Savings Accounts, Money Transfers).  
- Clean complaint narratives by lowercasing, removing special characters, and normalizing text.  
- Save the cleaned dataset for downstream processing.

**Task 2 – Text Chunking, Embedding, and Vector Store Indexing**  
- Split long narratives into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`.  
- Generate dense vector embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model.  
- Store embeddings along with metadata (complaint ID, product, issue) in a persistent ChromaDB vector store.  
- Batch processing is implemented for efficiency, with error handling for model loading and database operations.

---

## Usage

### 1. Preprocessing
```python
from src.preprocessing import load_data, clean_text, filter_products
import pandas as pd

df = load_data("../data/raw/complaints.csv")
df_filtered = filter_products(df, ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"])
````

### 2. Chunking

```python
from src.chunking import ChunkManager

chunker = ChunkManager(df_filtered, chunk_size=500, chunk_overlap=50)
df_chunks = chunker.create_chunks()
df_chunks.to_csv("../data/processed/chunks.csv", index=False)
```

### 3. Vector Store

```python
from src.vector_store import VectorStoreManager

vsm = VectorStoreManager(df_chunks)
vsm.build_vector_store(batch_size=128)
```

---

## Notes

* All core logic is in `src/` modules to facilitate reuse and clarity.
* Error handling is included for file I/O, model loading, and ChromaDB operations.
* The notebook `Task1_Task2_analysis.ipynb` orchestrates the pipeline and includes visualizations for EDA and narrative length analysis.

---


