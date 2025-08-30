# app/llm_interpreter.py
import os, json
from openai import OpenAI
from .query_engine import BAUniversalQueryEngine

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "master_table.csv")
engine = BAUniversalQueryEngine(DATA_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INTERPRET_SYS = """You are a careful information extractor.
You receive a user request and a list of real column names from a master table.
Your job: map the user request to a small JSON of filters ONLY using the provided columns.
- Use exact column names from the list.
- If unsure, return {}.
Return ONLY valid JSON, no prose.
"""

ANSWER_SYS = """You are BAIT—an experienced Business Analyst.
Write a concise human answer based ONLY on the provided EVIDENCE rows (csv slices).
Guardrails:
- Do NOT invent facts not present in EVIDENCE.
- Synthesize and structure as guidance (short paragraphs + bullet 'Next actions' if present).
- If EVIDENCE is empty, ask for one clarifying question, no hallucinations.
Also return a short 'evidence' list (e.g., row indices or brief refs) to prove grounding.
Output JSON:
{"answer": "...", "evidence_refs": ["...","..."]}
"""

def _chat(model, sys, user_msg, **params):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},{"role":"user","content":user_msg}],
        temperature=0.2,
        **params
    )
    return resp.choices[0].message.content.strip()

def interpret_query(user_input: str):
    # A) извлечи филтри
    cols = engine.df.columns.tolist()
    interpret_user = f"User request: {user_input}\nColumns: {json.dumps(cols)}\nReturn JSON of filters."
    raw = _chat("gpt-4o-mini", INTERPRET_SYS, interpret_user, max_tokens=300)

    try:
        filters = json.loads(raw)
        if not isinstance(filters, dict):
            filters = {}
    except Exception:
        filters = {}

    # B) извлечи доказателства от таблицата
    _, evidence_df = engine.query(**filters)

    # C) отговор на човек, но само от evidence
    if evidence_df.empty:
        evidence_snippet = "[]"
    else:
        # малък срез от редовете като текст за LLM
        evidence_snippet = evidence_df.to_csv(index=False)

    answer_user = f"EVIDENCE (csv):\n{evidence_snippet}\n\nUser request: {user_input}\n"
    raw_answer = _chat("gpt-4o-mini", ANSWER_SYS, answer_user, max_tokens=800)

    # върни чист текст към фронта; пазим и JSON за евентуален dev режим
    try:
        obj = json.loads(raw_answer)
        answer = obj.get("answer", "").strip() or raw_answer
        evidence_refs = obj.get("evidence_refs", [])
    except Exception:
        answer = raw_answer
        evidence_refs = []

    # dev hint – може да логнеш evidence_refs
    return answer