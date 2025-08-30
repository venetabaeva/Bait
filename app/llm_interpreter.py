import os, json
from openai import OpenAI
from app.query_engine import BAUniversalQueryEngine

# зареди таблицата
ENGINE = BAUniversalQueryEngine(master_table_path="app/data/master_table.csv")

def interpret_query(user_input: str) -> str:
    if not user_input:
        return "Please enter a question or description."

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    factors = ENGINE.get_all_factors()

    sys_prompt = f"""
You are a Business Analysis assistant.
Map the user's message to the most relevant factor values from this list of columns (factors):
{factors}

Return ONLY a compact JSON object where keys are factor/column names and values are the chosen values.
If unsure about a factor, omit it. Example: {{"Persona": "Sponsor", "Condition": "High risk"}}
Do NOT add explanations outside JSON.
"""

    # 1) LLM → JSON мапинг фактори
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user_input}
        ],
        max_tokens=300
    )

    raw = resp.choices[0].message.content.strip()
    try:
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            mapping = {}
    except Exception:
        mapping = {}  # ако не е валиден JSON — продължаваме с reasoning без твърдо филтриране

    # 2) филтриране на master-а
    rows = ENGINE.query(**mapping)

    # 3) човешко резюме на правилата/насоките
    summary = ENGINE.summarize_rows(rows)

    # 4) финален отговор (малко reasoning + summary)
    final_prompt = f"""
User input: {user_input}
Mapped factors: {json.dumps(mapping, ensure_ascii=False)}

Master table summary (for the mapped subset):
{summary}

Write a concise, human response (5–10 sentences). Give practical recommendations.
Do not list the whole table. If no exact match exists, reason from the closest factors.
End with 2–3 bullet points of next actions.
"""
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a senior BA advisor."},
                  {"role":"user","content":final_prompt}],
        max_tokens=350,
        temperature=0.4,
    ).choices[0].message.content.strip()

    return final