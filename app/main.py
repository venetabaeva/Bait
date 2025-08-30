import os
import uuid
from typing import Dict, Deque
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.llm_interpreter import interpret_query

# -------------------------
# FastAPI и CORS
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ако знаеш домейна, сложи конкретния URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# статиката от ui/static достъпна на /static
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# -------------------------
# Проста памет по сесия (в RAM)
# -------------------------
# { sid: deque([{"role":"user"/"assistant", "content":"..."}], maxlen=10) }
SESSIONS: Dict[str, Deque[dict]] = {}

def get_or_create_sid(request: Request) -> str:
    sid = request.cookies.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

# -------------------------
# Начална страница
# -------------------------
@app.get("/")
def serve_homepage(request: Request):
    # подаваме cookie ако липсва
    sid = get_or_create_sid(request)
    resp = FileResponse(os.path.join("ui", "index.html"))
    resp.set_cookie("sid", sid, httponly=False, samesite="Lax")
    return resp

# -------------------------
# Чат endpoint
# -------------------------
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = (data or {}).get("message", "").strip()
        if not user_message:
            return JSONResponse({"response": "Празно съобщение."}, status_code=200)

        # вземи/създай sid и историята
        sid = request.cookies.get("sid") or data.get("session_id")
        if not sid:
            sid = get_or_create_sid(request)

        history = SESSIONS.get(sid)
        if history is None:
            history = deque(maxlen=10)
            SESSIONS[sid] = history

        # детекция на език (много проста)
        def detect_lang(txt: str) -> str:
            return "bg" if any("\u0400" <= ch <= "\u04FF" for ch in txt) else "en"

        lang = detect_lang(user_message)

        # извикай интерпретатора (LLM + правила от таблицата)
        reply_text = interpret_query(
            user_input=user_message,
            history=list(history),   # предаваме досегашния контекст
            lang=lang,               # "bg" или "en"
            data_path=os.path.join(os.path.dirname(_file_), "data", "master_table.csv"),
        )

        # обнови историята
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply_text})

        # върни резултата и cookie
        resp = JSONResponse({"response": reply_text})
        resp.set_cookie("sid", sid, httponly=False, samesite="Lax")
        return resp

    except Exception as e:
        # показваме ясна грешка, за да се вижда в UI
        return JSONResponse({"response": f"Server error: {e}"}, status_code=500)