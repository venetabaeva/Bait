import os, json, hashlib
from openai import OpenAI
from dotenv import load_dotenv
from .query_engine import BAUniversalQueryEngine

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(_file_), "data", "master_table.csv")
QE = BAUniversalQueryEngine(DATA_PATH)

def _hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]

DATASET_VERSION = _hash(DATA_PATH)

def _client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def interpret_query(user_input: str) -> dict:
    """
    Strictly grounded: we only answer from table rows.
    Steps: detect Activity -> fetch rows -> compose template -> optional LLM rephrase (no new facts).
    """
    # 1) detect Activity text via LLM (minimal use)
    factors = {}
    try:
        sys = (
            "Extract a short 'Activity' phrase from the user's message. "
            "Return JSON like {\"Activity\": \"...\"}. No other fields."
        )
        r = _client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user_input}],
            temperature=0,
            max_tokens=60
        )
        raw = r.choices[0].message.content.strip()
        try:
            factors = json.loads(raw)
            if not isinstance(factors, dict):
                factors = {}
        except json.JSONDecodeError:
            factors = {}
    except Exception as e:
        return {"answer": f"LLM error: {e}", "factors": {}, "matched_count": 0,
                "matched_preview": [], "dataset_version": DATASET_VERSION}

    activity = (factors.get("Activity") or "").strip()
    # 2) query table (robust contains on Activity)
    try:
        rows = QE.query_contains(Activity=activity) if activity else QE.df.head(0)
    except Exception as e:
        return {"answer": f"Query error: {e}", "factors": {"Activity": activity},
                "matched_count": 0, "matched_preview": [], "dataset_version": DATASET_VERSION}

    matched_count = len(rows)
    preview = rows.head(5).to_dict(orient="records")

    if matched_count == 0:
        return {
            "answer": (
                "I couldn’t find a direct rule for this activity in the master table. "
                "Please try a more specific phrasing for the activity that appears in the table."
            ),
            "factors": {"Activity": activity},
            "matched_count": 0,
            "matched_preview": [],
            "dataset_version": DATASET_VERSION
        }

    # 3) deterministic template from columns
    # Try common column names; degrade gracefully if some are missing
    cols = [c.lower() for c in QE.df.columns]
    def pick(*names):
        for n in names:
            for c in QE.df.columns:
                if c.lower() == n.lower():
                    return c
        return None

    c_activity    = pick("Activity")
    c_stakeholder = pick("Stakeholder","Stakeholders","Role")
    c_expectation = pick("Expectation","Expected from stakeholder","Expectations")
    c_next        = pick("Next actions","Next steps","Actions")

    # build bullet facts
    stakeholders = []
    exp_lines = []
    next_lines = []

    for _, rrow in rows.iterrows():
        if c_stakeholder and str(rrow.get(c_stakeholder,"")).strip():
            stakeholders.append(str(rrow[c_stakeholder]).strip())
        if c_stakeholder and c_expectation:
            s = str(rrow.get(c_stakeholder,"")).strip()
            e = str(rrow.get(c_expectation,"")).strip()
            if s or e:
                exp_lines.append(f"- {s}: {e}".strip(": "))
        if c_next and str(rrow.get(c_next,"")).strip():
            nxt = str(rrow[c_next]).strip()
            next_lines.append(f"- {nxt}")

    stakeholders = sorted(set([s for s in stakeholders if s]))
    exp_lines    = [x for x in exp_lines if x]
    next_lines   = [x for x in next_lines if x]

    template_answer = []
    if c_activity and activity:
        template_answer.append(f"Activity: {activity}")
    if stakeholders:
        template_answer.append("Stakeholders involved:")
        template_answer.extend(exp_lines if exp_lines else [f"- {s}" for s in stakeholders])
    if next_lines:
        template_answer.append("Next actions:")
        template_answer.extend(next_lines)

    plain = "\n".join(template_answer).strip()

    # 4) optional rephrase: make it human, but DO NOT add facts
    final_text = plain
    try:
        prompt = (
            "Rewrite the following bullets into a concise human advisory answer. "
            "Do NOT add any new facts or stakeholders. Only rephrase what is present.\n\n"
            f"{plain}"
        )
        r2 = _client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You must not introduce information not present in user content."},
                      {"role":"user","content":prompt}],
            temperature=0,
            max_tokens=350
        )
        final_text = r2.choices[0].message.content.strip() or plain
    except Exception:
        pass

    return {
        "answer": final_text,
        "factors": {"Activity": activity},
        "matched_count": matched_count,
        "matched_preview": preview[:3],
        "dataset_version": DATASET_VERSION
    }