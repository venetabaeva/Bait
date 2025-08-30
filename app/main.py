import os
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.llm_interpreter import interpret_query, DATA_PATH

app = FastAPI()

# CORS – разрешаваме фронтенда да вика /chat
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можеш да ограничиш към твоя домейн
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# статични файлове (логото и index.html)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join("ui", "index.html"))

# Проста памет на сесия (в паметта на процеса)
CHAT_HISTORY: List[Dict[str, str]] = []


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message: str = data.get("message", "")
        lang: str = data.get("lang", "bg")

        # добавяме входа в историята
        CHAT_HISTORY.append({"role": "user", "content": user_message})

        # извикваме интерпретатора
        reply = interpret_query(
            user_input=user_message,
            history=CHAT_HISTORY,
            lang=lang,
            data_path=DATA_PATH,  # явен път
        )

        # добавяме отговора на модела в историята
        CHAT_HISTORY.append({"role": "assistant", "content": reply})

        return JSONResponse({"response": reply})
    except Exception as e:
        # връщаме 500 към UI с кратко съобщение
        return JSONResponse({"response": f"Server error: {e}"}, status_code=500)


# Локално стартиране (Render го игнорира, тъй като ползва Procfile)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))