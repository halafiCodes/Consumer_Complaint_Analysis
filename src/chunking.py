from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

class ChunkManager:
    def __init__(self, df, chunk_size=500, chunk_overlap=50):
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot create chunks.")
        self.df = df
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def create_chunks(self):
        all_chunks = []
        for _, row in self.df.iterrows():
            chunks = self.splitter.split_text(row['cleaned_narrative'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'complaint_id': row['Complaint ID'],
                    'product': row['Product'],
                    'issue': row['Issue'],
                    'chunk_index': i,
                    'text': chunk
                })
        df_chunks = pd.DataFrame(all_chunks)
        return df_chunks
