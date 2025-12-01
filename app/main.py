import os
import threading
import time
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
from embedding_utils import load_knowledge_base, build_embeddings, find_relevant_context
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import google.generativeai as genai

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

genai.configure(api_key="")

app = FastAPI(title="Chatbot API (Excel Upload)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use /data folder for Excel and embeddings
DATA_DIR = "../data"
KB_PATH = os.path.join(DATA_DIR, "knowledge.xlsx")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Watchdog Event Handler ---
class KnowledgeFileHandler(FileSystemEventHandler):
    def __init__(self, kb_path, emb_path):
        self.kb_path = kb_path
        self.emb_path = emb_path
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith("knowledge.xlsx"):
            now = time.time()
            if now - self.last_modified > 3:  # debounce rapid saves
                print("📄 Detected Excel file update — rebuilding embeddings...")
                df = load_knowledge_base(self.kb_path)
                build_embeddings(df, self.emb_path)
                self.last_modified = now

def start_file_watcher(kb_path, emb_path):
    event_handler = KnowledgeFileHandler(kb_path, emb_path)
    observer = Observer()
    observer.schedule(event_handler, path=DATA_DIR, recursive=False)
    observer.start()
    print("👀 Watching for Excel file changes...")
    return observer

# --- Initialize ---
if os.path.exists(KB_PATH):
    df = load_knowledge_base(KB_PATH)
    build_embeddings(df, EMB_PATH)
else:
    print(f"⚠️ {KB_PATH} not found. Upload Excel via /upload first.")

observer = start_file_watcher(KB_PATH, EMB_PATH)

@app.on_event("shutdown")
def stop_watcher():
    observer.stop()
    observer.join()

@app.get("/")
async def root():
    return {"message": "Chatbot API with Groq + Excel Upload running 🚀"}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("message", {}).get("text", "")
    #query = data.get("message", "")
    if not query:
        return {"reply": "Please provide a question."}

    if not os.path.exists(EMB_PATH):
        return {"reply": "Knowledge base not found. Upload Excel via /upload."}

    context = find_relevant_context(query, embed_file=EMB_PATH)
    prompt = f"""
You are a helpful assistant. Use this context from the company's knowledge base to answer accurately.
If the context doesn’t contain an answer, say “I’m not sure about that.”

Context:
{context}

Question: {query}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        reply = f"Error from Groq: {e}"

    return {
        "replies": [
            {
                "text": reply
            }
        ]
            }
            
@app.post("/ask_gemini")
async def ask_gemini(request: Request):
    data = await request.json()

    query = data.get("message", {}).get("text", "")
    if not query:
        return {"reply": "Please provide a question."}

    if not os.path.exists(EMB_PATH):
        return {"reply": "Knowledge base not found. Upload Excel via /upload."}

    context = find_relevant_context(query, embed_file=EMB_PATH)

    # ---- PROMPT ----
    prompt = f"""
You are a precise and reliable assistant. You are given a context extracted from a trained knowledge base built from an Excel file that contains pairs of Questions and Answers.

Your job:
1. If the provided context contains a clear, relevant answer, respond ONLY using that answer.
2. Do NOT create or guess information that is not present in the context.
3. If the context is empty, irrelevant, or does not contain the answer, respond exactly with:
"For more info contact info@enestit.com"
4. Keep your answer short, direct, and strictly factual.

Context:
{context}

Question:
{query}

Provide the final answer now.
"""

    try:
        model = genai.GenerativeModel("models/gemini-pro-latest")
        response = model.generate_content(prompt)

        # --- SAFE EXTRACTION ---
        reply = None

        # Try direct text
        try:
            reply = response.text
        except:
            pass

        # Try candidate parts
        if not reply:
            try:
                reply = response.candidates[0].content.parts[0].text
            except:
                pass

        # Final fallback if Gemini returned empty
        if not reply or reply.strip() == "":
            reply = "For more info contact info@enestit.com"

    except Exception as e:
        reply = f"Error from Gemini: {e}"

    return {
        "replies": [{"text": reply}]
    }
    
@app.post("/ask_gemini_dynamic")
async def ask_gemini_dynamic(request: Request):
    """
    Gemini endpoint with dynamic system prompt, optional model selection,
    and similarity threshold filtering.
    
    Request body example:
    {
      "message": {
        "system_prompt": "You are a helpful assistant that answers only from knowledge base.",
        "text": "What is the warranty period?"
      },
      "model": "models/gemini-pro-latest"
    }
    """
    data = await request.json()
    message = data.get("message", {})

    system_prompt = message.get("system_prompt", "")
    query = message.get("text", "")

    model_name = data.get("model", "models/gemini-pro-latest")  # default model

    if not query:
        return {"reply": "Please provide a question."}

    if not os.path.exists(EMB_PATH):
        return {"reply": "Knowledge base not found. Upload Excel via /upload."}

    # --- Retrieve context from embeddings with similarity threshold ---
    context = find_relevant_context(query, embed_file=EMB_PATH, threshold=0.65)

    # If no relevant context, fallback
    if context.strip() == "":
        return {"replies": [{"text": "For more info contact info@enestit.com"}]}

    # --- Build final prompt ---
    prompt = f"""
{system_prompt}

Context:
{context}

User Question:
{query}

Provide the final answer now.
"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        # --- SAFE EXTRACTION ---
        reply = None
        try:
            reply = response.text
        except:
            pass

        if not reply:
            try:
                reply = response.candidates[0].content.parts[0].text
            except:
                pass

        # Final fallback if Gemini returned empty
        if not reply or reply.strip() == "":
            reply = "For more info contact info@enestit.com"

    except Exception as e:
        reply = f"Error from Gemini: {e}"

    return {"replies": [{"text": reply}]}

@app.get("/gemini_models")
def list_gemini_models():
    try:
        models = genai.list_models()
        return {"models": [m.name for m in models]}
    except Exception as e:
        return {"error": str(e)}

# --- Upload Excel Endpoint ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    path = KB_PATH
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Rebuild embeddings immediately
    df = load_knowledge_base(path)
    build_embeddings(df, EMB_PATH)
    return {"message": "File uploaded and embeddings rebuilt."}
