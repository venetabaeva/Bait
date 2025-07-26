from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.query_engine import BAUniversalQueryEngine
from app.llm_interpreter import interpret_query
import os

app = FastAPI()

# Serve UI
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/")
def read_index():
    return FileResponse("ui/index.html")

# Initialize Query Engine
query_engine = BAUniversalQueryEngine("app/data/master_table.csv")

@app.post("/query")
def ask_agent(user_query: str = Body(..., embed=True)):
    factors = query_engine.get_all_factors()
    interpreted = interpret_query(user_query, factors)

    if not interpreted:
        return {"answer": "I couldn't map your query to any known factors."}

    results = query_engine.query(**interpreted)
    if results.empty:
        return {"answer": f"No results found for: {interpreted}"}

    return {
        "query_factors": interpreted,
        "results": results.head(10).to_dict(orient="records")
    }
