import os
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.agents.chatbot_agent import ChatbotAgent
from backend.agents.news_agent import WebAgent
from backend.agents.sql_agent import SQLAgent
from backend.agents.analyst_agent import AnalystAgent
from backend.agents.notes_agent import NotesAgent
from backend.agents.code_review import CodeReviewerAgent  # make sure filename matches
from backend.agents.router_agent import RouterAgent


app = FastAPI(title="Multi-Agent API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("data", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

# Upload dir
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Agents
chatbot = ChatbotAgent()
news = WebAgent()
sql = SQLAgent()
analyst = AnalystAgent()
notes = NotesAgent()
code_review = CodeReviewerAgent()
router = RouterAgent()

# Schemas
class ChatReq(BaseModel):
    message: str

class NewsReq(BaseModel):
    question: str

class SqlReq(BaseModel):
    prompt: str

class AnalystReq(BaseModel):
    csv_text: Optional[str] = None
    ask: Optional[str] = None

class NotesReq(BaseModel):
    notes: Optional[str] = None
    prompt: Optional[str] = None
    audience: Optional[str] = "Team"
    tone: Optional[str] = "crisp, neutral, professional"
    max_len: Optional[int] = 350

# 🔥 FIXED: accept both code and text, plus optional context
class CodeReviewReq(BaseModel):
    code: Optional[str] = None
    text: Optional[str] = None
    context: Optional[str] = None
    



@app.get("/health")
def health():
    return {"ok": True}


def _with_timing(payload: dict, started_at: float):
    ms = int((time.time() - started_at) * 1000)
    payload.setdefault("_meta", {})["latency_ms"] = ms
    return payload


# ---------- Routes ----------

@app.post("/api/chatbot")
async def api_chatbot(b: ChatReq):
    t0 = time.time()
    try:
        out = await run_in_threadpool(chatbot.run, b.message)
        return _with_timing({"output": out}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in chatbot: {e}"}, t0)


@app.post("/api/webnews")
async def api_webnews(b: NewsReq):
    t0 = time.time()
    try:
        out = await run_in_threadpool(news.ask, b.question)
        return _with_timing({"output": out}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in news: {e}"}, t0)


@app.post("/api/sql")
async def api_sql(b: SqlReq):
    t0 = time.time()
    try:
        out = await run_in_threadpool(sql.run, b.prompt)
        return _with_timing({"output": out}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in SQL: {e}"}, t0)


@app.post("/api/analyst")
async def api_analyst(b: AnalystReq):
    t0 = time.time()
    try:
        ask = b.ask or "Load this CSV and summarize; then create a relevant chart."
        csv = b.csv_text or ""
        prompt = f"Here is a CSV:\n```csv\n{csv}\n```\n{ask}"
        out = await run_in_threadpool(analyst.run, prompt)

        img = getattr(analyst, "last_plot_path", None)
        web_img = (f"/{img}" if img and not img.startswith("/") else img)

        return _with_timing({"output": out, "image_path": web_img}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in analyst: {e}"}, t0)


@app.post("/api/analyst_file")
async def api_analyst_file(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        dest = UPLOAD_DIR / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        prompt = (
            f"Load this CSV from file path: {dest.as_posix()}\n"
            "Summarize shape, columns, dtypes, nulls, and key stats. "
            "Then create a suitable chart."
        )
        out = await run_in_threadpool(analyst.run, prompt)

        img = getattr(analyst, "last_plot_path", None)
        web_img = (f"/{img}" if img and not img.startswith("/") else img)

        return _with_timing({"output": out, "image_path": web_img}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in analyst_file: {e}"}, t0)


@app.post("/api/notes")
async def api_notes(b: NotesReq):
    t0 = time.time()
    try:
        text = b.notes or b.prompt or ""
        out = await run_in_threadpool(
            notes.summarize,
            text,
            b.audience or "Team",
            b.tone or "crisp, neutral, professional",
            b.max_len or 350,
        )
        return _with_timing({"output": out}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in NotesAgent: {e}"}, t0)


@app.post("/api/code_review")
async def api_code_review(b: CodeReviewReq):
    t0 = time.time()
    try:
        raw = b.code or b.text or ""
        if not raw.strip():
            return _with_timing({"output": "Error: No code provided."}, t0)

        out = await run_in_threadpool(code_review.review, raw, b.context)
        return _with_timing({"output": out}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in CodeReviewer: {e}"}, t0)

@app.post("/api/router")
async def api_router(b: ChatReq):
    t0 = time.time()
    try:
        out = await run_in_threadpool(router.run, b.message)
        return _with_timing({"output": out}, t0)
    except Exception as e:
        return _with_timing({"output": f"Error in RouterAgent: {e}"}, t0)