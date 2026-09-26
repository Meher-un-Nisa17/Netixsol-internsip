import sys
import os
from pathlib import Path
import uuid
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

# Fix path to load root modules correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

from rag.rag_pipeline import RealEstateRAGPipeline
from utils.edge_tts_integration import EdgeVoiceSynthesizer
from deepgram import DeepgramClient

# Initialize FastAPI App
app = FastAPI(title="Uzma Real Estate Voice Agent Production Backend")

# Enable CORS for Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio output directory exists and mount static files
AUDIO_OUTPUT_DIR = Path(os.getenv("AUDIO_OUTPUT_DIR", "data/audio_output"))
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(AUDIO_OUTPUT_DIR)), name="static")

# Initialize Core System Components Globally
print("[System Init]: Real Estate RAG pipeline will load per conversation...")
SESSION_LIMIT = max(1, int(os.getenv("SESSION_LIMIT", "256")))
SESSION_TTL_SECONDS = max(60, int(os.getenv("SESSION_TTL_SECONDS", "3600")))
session_pipelines = OrderedDict()
session_lock = threading.RLock()

def get_session_pipeline(session_id: str):
    now = time.monotonic()
    with session_lock:
        expired = [key for key, item in session_pipelines.items() if now - item["last_seen"] > SESSION_TTL_SECONDS]
        for key in expired:
            session_pipelines.pop(key, None)
        if session_id not in session_pipelines:
            session_pipelines[session_id] = {"pipeline": RealEstateRAGPipeline(), "last_seen": now, "lock": threading.RLock()}
        session_pipelines.move_to_end(session_id)
        while len(session_pipelines) > SESSION_LIMIT:
            session_pipelines.popitem(last=False)
        item = session_pipelines[session_id]
        item["last_seen"] = now
        return item

def handle_pipeline_turn(session, message: str):
    with session["lock"]:
        return session["pipeline"].handle_turn(message)

print("[System Init]: Loading Edge-TTS Voice Engine (ur-PK-UzmaNeural)...")
synthesizer = EdgeVoiceSynthesizer()

print("[System Init]: Initializing Deepgram Nova-3 STT Client...")
api_key = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(api_key=api_key) if api_key else None

# SQLite Database setup with WAL mode
def get_db():
    db_path = os.getenv("VISITS_API_DB_PATH", "visits.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_name TEXT,
            phone TEXT UNIQUE,
            email TEXT,
            target_area TEXT,
            city TEXT,
            max_budget REAL,
            meeting_time TEXT,
            event_id TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class VisitRequest(BaseModel):
    client_name: str
    phone: str
    email: Optional[str] = None
    target_area: Optional[str] = None
    city: Optional[str] = None
    max_budget: Optional[float] = None
    meeting_time: str
    event_id: Optional[str] = None

class RescheduleRequest(BaseModel):
    phone: str
    meeting_time: str
    event_id: Optional[str] = None

class CancelRequest(BaseModel):
    phone: str
    event_id: Optional[str] = None

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "service": "real-estate-voice-agent-backend"}

@app.post("/chat")
async def chat_endpoint(request: Request, req: ChatRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        session = get_session_pipeline(session_id)
        print("[Text Input] Received one caller turn.")
        # Pass text directly into the robust RAG pipeline
        text_response = await run_in_threadpool(handle_pipeline_turn, session, req.message)
        print("[Uzma Response] Text turn completed.")
        
        # Generate Edge-TTS Audio
        timestamp_str = uuid.uuid4().hex[:8]
        response_filename = f"response_{timestamp_str}.wav"
        response_path = AUDIO_OUTPUT_DIR / response_filename
        
        await synthesizer.synthesize_speech_async(text=text_response, output_path=str(response_path))
        public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/") or str(request.base_url).rstrip("/")
        audio_url = f"{public_base_url}/static/{response_filename}"
        
        return {
            "status": "success",
            "data": {
                "response": text_response,
                "audio_path": audio_url,
                "session_id": session_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-chat")
async def voice_chat_endpoint(request: Request, file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    try:
        if not deepgram_client:
            raise HTTPException(status_code=503, detail="Voice transcription is not configured")
        audio_bytes = await file.read()
        
        # 1. Transcribe via Deepgram Nova-3 API
        response = deepgram_client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-3",
            language="ur",
            smart_format=True
        )
        
        transcribed_text = response.results.channels[0].alternatives[0].transcript.strip()
        print("[Deepgram] Voice transcription completed.")
        
        if not transcribed_text:
            transcribed_text = "سلام"
            
        # 2. Process through this caller's isolated conversation state
        session_id = session_id or str(uuid.uuid4())
        session = get_session_pipeline(session_id)
        text_response = await run_in_threadpool(handle_pipeline_turn, session, transcribed_text)
        print("[Uzma Response] Voice turn completed.")
        
        # 3. Generate Edge-TTS Neural Audio Output
        timestamp_str = uuid.uuid4().hex[:8]
        response_filename = f"response_{timestamp_str}.wav"
        response_path = AUDIO_OUTPUT_DIR / response_filename
        
        await synthesizer.synthesize_speech_async(text=text_response, output_path=str(response_path))
        public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/") or str(request.base_url).rstrip("/")
        audio_url = f"{public_base_url}/static/{response_filename}"
        
        return {
            "status": "success",
            "data": {
                "transcribed_text": transcribed_text,
                "response": text_response,
                "audio_path": audio_url,
                "session_id": session_id
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Database endpoints...
@app.post("/api/save-visit")
def save_visit(visit: VisitRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO visits (client_name, phone, email, target_area, city, max_budget, meeting_time, event_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(phone) DO UPDATE SET
            client_name = excluded.client_name, email = excluded.email,
            target_area = excluded.target_area, city = excluded.city,
            max_budget = excluded.max_budget, meeting_time = excluded.meeting_time,
            event_id = excluded.event_id
    """, (visit.client_name, visit.phone, visit.email, visit.target_area, visit.city, visit.max_budget, visit.meeting_time, visit.event_id))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Visit saved"}

@app.post("/api/reschedule-visit")
def reschedule_visit(req: RescheduleRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE visits SET meeting_time = ?, event_id = COALESCE(?, event_id) WHERE phone = ?", (req.meeting_time, req.event_id, req.phone))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Rescheduled"}

@app.post("/api/cancel-visit")
def cancel_visit(req: CancelRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM visits WHERE phone = ?", (req.phone,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Cancelled"}
