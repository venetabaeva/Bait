from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from app.llm_interpreter import interpret_query

app = FastAPI()
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join("ui", "index.html"))

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    msg = data.get("message", "")
    try:
        reply = interpret_query(msg)
        return JSONResponse({"response": reply})
    except Exception as e:
        return JSONResponse({"response": f"Error: {e}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
