from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app.llm_interpreter import interpret_query

app = FastAPI()

# позволи заявките от браузъра
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# статични файлове: /static → ui/static
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# начална страница
@app.get("/")
def home():
    return FileResponse(os.path.join("ui", "index.html"))

# чат API
@app.post("/chat")
async def chat(req: Request):
    try:
        data = await req.json()
        user_msg = data.get("message", "").strip()
        reply = interpret_query(user_msg)
        return JSONResponse({"response": reply})
    except Exception as e:
        # не печатай CSV/дебъг в отговора
        return JSONResponse({"response": f"Server error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))