# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app.llm_interpreter import interpret_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""], allow_methods=[""], allow_headers=["*"]
)

# Static UI
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join("ui", "index.html"))

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    msg = data.get("message","").strip()
    try:
        reply = interpret_query(msg)
        return JSONResponse({"response": reply})
    except Exception as e:
        return JSONResponse({"response": f"Error: {e}"}, status_code=500)

if __name__ == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)