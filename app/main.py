import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# your interpreter (must exist at app/llm_interpreter.py)
from .llm_interpreter import interpret_query

app = FastAPI(title="BAIT")

# allow browser requests (Render + local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve /static/* from ui/static (logo, css, etc.)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")


@app.get("/health")
def health():
    return PlainTextResponse("ok")


# serve the UI
@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join("ui", "index.html"))


# chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    msg = (data.get("message") or "").strip()
    debug = bool(data.get("debug")) or ("debug" in request.query_params)

    try:
        result = interpret_query(msg)  # returns dict with keys: answer, factors, matched_count, ...
        if debug:
            return JSONResponse(result)
        return JSONResponse({"response": result.get("answer", "")})
    except Exception as e:
        # surface error but keep 500 for visibility in Render logs
        return JSONResponse({"response": f"Error: {str(e)}"}, status_code=500)


if _name_ == "_main_":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Render sets PORT
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)