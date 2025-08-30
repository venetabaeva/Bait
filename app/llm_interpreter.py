import os
import json
from typing import List, Dict, Deque, Tuple
from collections import deque

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# === Път до master таблицата (остави така) ===
DATA_PATH = os.path.join(os.path.dirname(_file_), "data", "master_table.csv")

# === Настройки за история/контекст на чата (в рамките на една сесия) ===
MAX_TURNS = 6  # колко последни потребител/бот реплики да пазим

# === Зареждане на API ключа ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Модел
OPENAI_MODEL = "gpt-4o-mini"

# ------------------ Помощни функции върху таблицата ------------------

def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.fillna("")
    return df

def top_n_matches(df: pd.DataFrame, user_text: str, top_n: int = 5) -> List[Dict[str, str]]:
    """
    Много прост матчер: търси по колони Activity, Stakeholders, ExpectedActions, NextSteps, Risks, Mitigations
    и връща топ N реда, които имат най-много съвпадащи думи (casual match).
    """
    cols = [c for c in df.columns if c.lower() in {
        "activity", "stakeholders", "expectedactions", "nextsteps", "risks", "mitigations"
    }]
    if not cols:
        cols = list(df.columns)

    user_tokens = set(t.lower() for t in user_text.split())
    scored: List[Tuple[int, Dict[str, str]]] = []

    for _, row in df.iterrows():
        text = " ".join(str(row[c]) for c in cols)
        row_tokens = set(t.lower() for t in text.split())
        score = len(user_tokens & row_tokens)
        scored.append((score, {c: str(row[c]) for c in df.columns}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_n]]

def compress_rows(rows: List[Dict[str, str]], lang: str) -> str:
    """
    Прави кратък „facts blob“ за LLM — само фактологията от таблицата.
    """
    lines = []
    for r in rows:
        if lang.startswith("bg"):
            lines.append(
                f"* Дейност: {r.get('Activity','')}\n"
                f"  - Заинтересовани: {r.get('Stakeholders','')}\n"
                f"  - Очаквани действия: {r.get('ExpectedActions','')}\n"
                f"  - Следващи стъпки: {r.get('NextSteps','')}\n"
                f"  - Рискове: {r.get('Risks','')}\n"
                f"  - Митигиране: {r.get('Mitigations','')}\n"
            )
        else:
            lines.append(
                f"* Activity: {r.get('Activity','')}\n"
                f"  - Stakeholders: {r.get('Stakeholders','')}\n"
                f"  - Expected actions: {r.get('ExpectedActions','')}\n"
                f"  - Next steps: {r.get('NextSteps','')}\n"
                f"  - Risks: {r.get('Risks','')}\n"
                f"  - Mitigations: {r.get('Mitigations','')}\n"
            )
    blob = "\n".join(lines)
    # ограничаваме дължината
    return blob[:3000]

# ------------------ System prompt (БЕЗ уточняващи въпроси) ------------------

def system_prompt(lang: str, rules_blob: str) -> str:
    """
    Инструкции към модела: отговаряй директно, човешки, без да задаваш въпроси за уточняване.
    Базиран единствено на предоставените правила (таблични факти).
    """
    lang_hint = "Bulgarian" if (lang or "").lower().startswith("bg") else "English"
    return f"""
You are BAIT Advisor.

Answer STRICTLY and ONLY based on the factual content under 'RULES'.
Do NOT invent stakeholders, actions, or details that are not present in the rules.
Write a fluent, natural, human-style answer in {lang_hint}.
Never ask the user clarifying questions. Even if the input is vague, provide your best direct guidance relying only on the rules.
If something is missing in the rules and truly blocks a concrete step, state the gap briefly and still provide next steps that are justified by the rules.

RULES (tabular facts, already extracted):
{rules_blob}
""".strip()

# ------------------ Главна функция, извиквана от бекенда ------------------

def interpret_query(
    user_input: str,
    history: List[Dict[str, str]] | None = None,
    lang: str | None = "bg",
    data_path: str | None = None,
) -> str:
    """
    Намира релевантни редове от таблицата и генерира директен човешки отговор, БЕЗ уточняващи въпроси.
    """
    # 1) Зареждаме таблицата
    path = data_path or DATA_PATH
    df = load_table(path)

    # 2) Търсим най-релевантни редове
    matches = top_n_matches(df, user_input, top_n=5)
    rules = compress_rows(matches, lang or "bg")

    # 3) Сглобяваме съобщения за модела (история + система + потребител)
    sys_text = system_prompt(lang or "bg", rules)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_text}]

    # добавяме кратка история (ако има)
    trimmed: Deque[Dict[str, str]] = deque(maxlen=MAX_TURNS)
    if history:
        for m in history[-MAX_TURNS:]:
            # пазим само типичните полета role/content
            if "role" in m and "content" in m:
                trimmed.append({"role": m["role"], "content": m["content"]})
    msgs.extend(list(trimmed))

    # текущото потребителско съобщение
    msgs.append({"role": "user", "content": user_input})

    # 4) Извикваме модела (без да му позволяваме да отговаря на друг език)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.2,      # по-детерминирано
            max_tokens=700,
        )
        answer = resp.choices[0].message.content.strip()
        return answer
    except Exception as e:
        # безопасна грешка към UI
        return f"LLM error: {e}"