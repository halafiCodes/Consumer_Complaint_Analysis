import uuid
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

class VectorStoreManager:
    def __init__(self, df_chunks: pd.DataFrame, vector_store_path="../vector_store", collection_name="complaints"):
        if df_chunks.empty:
            raise ValueError("Vector store build aborted because df_chunks is empty.")
        self.df_chunks = df_chunks
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.model = self._load_model()
        self.collection = self._init_chromadb()

    def _load_model(self):
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

    def _init_chromadb(self):
        try:
            client = chromadb.PersistentClient(path=self.vector_store_path)
            collection = client.get_or_create_collection(name=self.collection_name)
            return collection
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

    def build_vector_store(self, batch_size=128):
        all_texts = self.df_chunks["text"].tolist()
        all_metadatas = self.df_chunks[["complaint_id", "product", "issue"]].to_dict('records')
        all_ids = [str(uuid.uuid4()) for _ in range(len(self.df_chunks))]

        if not all_texts:
            raise ValueError("No texts available for embedding and storage.")

        print(f"Starting batch processing for {len(all_texts)} chunks...")

        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            batch_metadatas = all_metadatas[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]

            try:
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False).tolist()
            except Exception as e:
                raise RuntimeError(f"Embedding batch starting at index {i} failed: {e}") from e

            try:
                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                raise RuntimeError(f"ChromaDB add failed for batch starting at index {i}: {e}") from e

            if i % (batch_size * 5) == 0:
                print(f"Processed {i + len(batch_texts)} / {len(all_texts)} chunks...")

        print("✅ Vector store saved successfully with batching!")
