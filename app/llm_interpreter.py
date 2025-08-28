import os, json
from dotenv import load_dotenv
from openai import OpenAI
from app.query_engine import BAUniversalQueryEngine

load_dotenv()

# пътят ти според снимките
MASTER_PATH = "app/data/master_table.csv"

engine = BAUniversalQueryEngine(MASTER_PATH)
FACTORS = engine.get_all_factors()

def interpret_query(user_input: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Server misconfigured: missing OPENAI_API_KEY."

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a BA Advisor Agent. The master table has the following factor columns: "
        + ", ".join(FACTORS)
        + ".\nUser query: " + user_input + "\n"
        "Return ONLY a JSON object of factor->value pairs inferred from the user query. "
        "If unsure, return {}."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # опит за JSON
        try:
            filters = json.loads(text)
            if not isinstance(filters, dict):
                filters = {}
        except json.JSONDecodeError:
            filters = {}

        df = engine.query(**filters)
        if df.empty:
            return "No relevant matches found."

        return df.to_string(index=False)
    except Exception as e:
        return f"LLM error: {e}"
