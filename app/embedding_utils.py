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
    h = (h * 4)[:256]
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
# Find Best Matching Answer with Similarity Threshold
# ---------------------------------------------------------
def find_relevant_context(query, k=3, embed_file="embeddings.pkl", threshold=0.65):
    """
    Returns the top answer(s) if similarity is above threshold.
    If no answer passes threshold, returns empty string.
    """
    if not os.path.exists(embed_file):
        raise FileNotFoundError("No embeddings found. Run build_embeddings() first.")

    index, df = pickle.load(open(embed_file, "rb"))
    q_emb = create_text_embedding(query)

    # Query FAISS
    D, I = index.search(np.array([q_emb]), k)

    # FAISS returns squared L2 distances; convert to similarity (roughly)
    # similarity = 1 / (1 + distance) → higher is better
    top_answers = []
    for dist, idx in zip(D[0], I[0]):
        similarity = 1 / (1 + dist)  # simple conversion
        if similarity >= threshold:
            top_answers.append(df.iloc[idx]["Answer"])

    # Return top answers concatenated or empty string if below threshold
    return "\n\n".join(top_answers) if top_answers else ""
