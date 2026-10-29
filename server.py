"""
FastAPI Web Backend for Saarthi AI
Exposes REST endpoints for real-time neural intent inference, system metrics, and notes management.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import psutil

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from saarthi.core.intent_classifier import IntentClassifier
from saarthi.core.executor import ActionExecutor
from saarthi.config import METRICS_PLOT_FILE, NOTES_FILE, MODEL_FILE

# Initialize FastAPI app
app = FastAPI(
    title="Saarthi AI - Web Backend",
    description="Deep Learning Voice & Multimodal Desktop Assistant Backend",
    version="1.0.0"
)

# Enable CORS for local cross-device demonstrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deep Learning Engine
print("[FastAPI] Initializing Saarthi Neural Intent Classifier...")
classifier = IntentClassifier()
executor = ActionExecutor(classifier)

# Web Assets Paths
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Request Schemas
class ChatRequest(BaseModel):
    query: str

class NoteRequest(BaseModel):
    note: str

# API Endpoints
@app.get("/")
async def serve_index():
    """Serves the main single-page web assistant console."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web console frontend not found.")
    return FileResponse(str(index_file))

@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming model status."""
    return {
        "status": "online",
        "assistant_name": "Saarthi AI",
        "model_loaded": classifier.is_loaded,
        "classes_count": len(classifier.label_encoder.classes_) if classifier.label_encoder else 0
    }

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    """
    Core AI Chat endpoint:
    Performs Bi-LSTM intent classification and executes corresponding skill.
    """
    query = request.query.strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Empty query received."})

    result = executor.process_query(query)
    return {
        "success": True,
        "query": query,
        "response_text": result["response_text"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "top_k": result.get("top_k", []),
        "action_taken": result.get("action_taken", "")
    }

@app.get("/api/system-status")
async def get_system_telemetry():
    """Fetches real-time CPU, RAM, and Battery telemetry for live dashboard gauges."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()

    battery_data = {
        "percent": battery.percent if battery else 100,
        "plugged": battery.power_plugged if battery else True,
        "has_battery": battery is not None
    }

    return {
        "cpu_percent": cpu_percent,
        "ram_percent": memory.percent,
        "ram_used_gb": round((memory.total - memory.available) / (1024 ** 3), 2),
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "ram_available_gb": round(memory.available / (1024 ** 3), 2),
        "battery": battery_data
    }

@app.get("/api/notes")
async def get_notes():
    """Retrieves all stored notes."""
    if not NOTES_FILE.exists():
        return {"notes": []}
    
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return {"notes": lines}

@app.post("/api/notes")
async def add_note(request: NoteRequest):
    """Appends a new note."""
    from saarthi.skills import productivity_skills
    msg = productivity_skills.take_note(request.note)
    return {"message": msg}

@app.delete("/api/notes")
async def clear_notes():
    """Clears all notes."""
    from saarthi.skills import productivity_skills
    msg = productivity_skills.clear_notes()
    return {"message": msg}

@app.get("/api/metrics-image")
async def get_metrics_image():
    """Serves the generated training metrics image for web display."""
    if not METRICS_PLOT_FILE.exists():
        raise HTTPException(status_code=404, detail="Metrics plot not generated yet.")
    return FileResponse(str(METRICS_PLOT_FILE), media_type="image/png")

@app.get("/api/model-info")
async def get_model_info():
    """Returns deep learning model metadata for academic presentation."""
    if not classifier.is_loaded:
        return {"status": "Model not loaded"}

    return {
        "architecture": "Bidirectional LSTM (Bi-LSTM)",
        "embedding_dim": 64,
        "max_sequence_length": 18,
        "classes_count": len(classifier.label_encoder.classes_),
        "classes": list(classifier.label_encoder.classes_),
        "framework": "TensorFlow 2.21 / Keras 3",
        "vocabulary_size": len(classifier.tokenizer.word_index) if classifier.tokenizer else 0
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("   SAARTHI AI - FASTAPI WEB DASHBOARD SERVER")
    print("   Running on http://127.0.0.1:8000 (Local Machine)")
    print("   Present to your teacher by opening http://localhost:8000")
    print("=" * 65 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
