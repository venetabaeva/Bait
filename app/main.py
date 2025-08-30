import os
import uuid
from collections import deque
from typing import Deque, Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.llm_interpreter import interpret_query

# ---------------------------
# In-memory сесии: sid -> deque от съобщения (роля/контент)
# ---------------------------
MAX_HISTORY = 8  # колко последни реплики пазим
SESSIONS: Dict[str, Deque[Dict[str, str]]] = {}

app = FastAPI()

# Разреши фронтенда (работим same-origin, но нека е широко)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# статична папка и UI
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join("ui", "index.html"))

@app.get("/healthz")
def health():
    return {"ok": True}

def _get_or_create_sid(req: Request, resp: Response) -> str:
    sid = req.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        # cookie 7 дни
        resp.set_cookie("sid", sid, httponly=False, samesite="lax", max_age=7*24*3600)
    if sid not in SESSIONS:
        SESSIONS[sid] = deque(maxlen=MAX_HISTORY)
    return sid

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    user_msg = (body or {}).get("message", "").strip()
    if not user_msg:
        return JSONResponse({"response": "Празно съобщение."}, status_code=200)

    resp = Response(media_type="application/json")
    sid = _get_or_create_sid(req, resp)

    # вземи текущата история за тази сесия
    history: Deque[Dict[str, str]] = SESSIONS[sid]

    try:
        answer_text = interpret_query(user_msg, list(history))  # подаваме историята
        # обнови историята
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer_text})
        return JSONResponse({"response": answer_text})
    except Exception as e:
        return JSONResponse({"response": f"Грешка: {e}"}, status_code=500)

# локално стартиране (Render не ползва това)
if _name_ == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)