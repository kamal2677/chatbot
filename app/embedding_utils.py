import os
import pandas as pd
import numpy as np
import faiss
import pickle
import hashlib

# ---------------------------------------------------------
# Load Excel Knowledge Base
# ---------------------------------------------------------
def load_knowledge_base(path="knowledge.xlsx"):
    df = pd.read_excel(path)
    df = df.fillna("")
    if "Question" not in df.columns or "Answer" not in df.columns:
        raise ValueError("Excel must contain 'Question' and 'Answer' columns.")
    return df


# ---------------------------------------------------------
# Create Text Embedding (Fixed – No More Errors)
# ---------------------------------------------------------
def create_text_embedding(text):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Expand the hash safely to 256 characters for 32-dim embedding
    h = (h * 4)[:256]  # 64 × 4 = 256 chars

    # Convert every 8 hex chars → integer
    vector = [int(h[i:i+8], 16) % 5000 for i in range(0, 256, 8)]

    return np.array(vector, dtype=np.float32)


# ---------------------------------------------------------
# Build Embeddings + FAISS Index
# ---------------------------------------------------------
def build_embeddings(df, save_path="embeddings.pkl"):
    print("🔄 Rebuilding embeddings...")

    questions = df["Question"].tolist()
    embeddings = [create_text_embedding(q) for q in questions]

    embeddings = np.vstack(embeddings).astype(np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index + dataframe together
    pickle.dump((index, df), open(save_path, "wb"))

    print(f"✅ Embeddings rebuilt and saved to {save_path}")


# ---------------------------------------------------------
# Find Best Matching Answer
# ---------------------------------------------------------
def find_relevant_context(query, k=3, embed_file="embeddings.pkl"):
    if not os.path.exists(embed_file):
        raise FileNotFoundError("No embeddings found. Run build_embeddings() first.")

    # Load stored FAISS index and dataframe
    index, df = pickle.load(open(embed_file, "rb"))

    # Create embedding for query
    q_emb = create_text_embedding(query)

    # Query FAISS
    D, I = index.search(np.array([q_emb]), k)

    # Return top-k answers
    return "\n\n".join(df.iloc[i]["Answer"] for i in I[0])
