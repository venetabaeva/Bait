from fastapi import FastAPI, Body
from app.query_engine import BAUniversalQueryEngine
from app.llm_interpreter import interpret_query

app = FastAPI()

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

    response = {
        "query_factors": interpreted,
        "results": results.head(10).to_dict(orient="records")
    }
    return response

@app.get("/")
def root():
    return {"message": "BA Advisor Agent is running."}
