print("STARTING SCRIPT...")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

print("Loading PDFs...")

loader = PyPDFDirectoryLoader("data")
docs = loader.load()

print(f"Loaded {len(docs)} pages")

print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = splitter.split_documents(docs)

print(f"Created {len(documents)} chunks")

print("Loading embedding model...")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings... (this may take time ⏳)")

texts = [doc.page_content for doc in documents]
embeddings = embed_model.encode(texts)
faiss.normalize_L2(embeddings)

print("Building FAISS index...")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings))

print("Saving vector store...")

import os
os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Medical knowledge base created successfully!")