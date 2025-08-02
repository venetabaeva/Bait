from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

# Import interpreter (OpenAI + dataset logic)
from app.llm_interpreter import interpret_query

app = FastAPI()

# Serve UI files
app.mount("/static", StaticFiles(directory="ui"), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join("ui", "index.html"))

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    
    try:
        response = interpret_query(user_message)
        return JSONResponse({"response": response})
    except Exception as e:
        return JSONResponse({"response": f"Error: {str(e)}"})

If __name__=="__main__":
	import unvicorn
	unvicorn.run(app, host="0.0.0.0", port=8000)
