from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app.llm_interpreter import interpret_query

app = FastAPI()

# Разрешаваме заявки от браузъра (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # може да сложиш и конкретен домейн вместо "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статични файлове (CSS, JS, картинки и т.н.)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# Главната страница
@app.get("/")
def home():
    return FileResponse(os.path.join("ui", "index.html"))

# Чат ендпойнт
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    msg = data.get("message", "")
    try:
        reply = interpret_query(msg)
        return JSONResponse({"response": reply})
    except Exception as e:
        return JSONResponse({"response": f"Error: {e}"}, status_code=500)

# Стартиране на uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
