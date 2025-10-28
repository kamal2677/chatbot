import os
import pandas as pd
import numpy as np
import faiss
import pickle
import hashlib

def load_knowledge_base(path="knowledge.xlsx"):
    df = pd.read_excel(path)
    df = df.fillna("")
    if "Question" not in df.columns or "Answer" not in df.columns:
        raise ValueError("Excel must contain 'Question' and 'Answer' columns.")
    return df

def create_text_embedding(text):
    # Simulated embedding via hashing
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return np.array([int(h[i:i+8], 16) % 1000 for i in range(0, 64, 8)], dtype=np.float32)

def build_embeddings(df, save_path="embeddings.pkl"):
    print("🔄 Rebuilding embeddings...")
    texts = (df["Question"] + " " + df["Answer"]).tolist()
    embeddings = [create_text_embedding(text) for text in texts]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))
    pickle.dump((index, df), open(save_path, "wb"))
    print(f"✅ Embeddings rebuilt and saved to {save_path}")

def find_relevant_context(query, k=3, embed_file="embeddings.pkl"):
    if not os.path.exists(embed_file):
        raise FileNotFoundError("No embeddings found. Run build_embeddings() first.")
    index, df = pickle.load(open(embed_file, "rb"))
    q_emb = create_text_embedding(query)
    D, I = index.search(np.array([q_emb]), k)
    return "\n\n".join(df.iloc[i]["Answer"] for i in I[0])
