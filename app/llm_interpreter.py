import os
import json
import pandas as pd
from typing import List, Dict
from openai import OpenAI

# Модел – можеш да смениш на по-силен при нужда
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# Четене и филтриране на master_table.csv
# -------------------------------------------------
REQUIRED_COLS = [
    # коригирай според твоите колони
    "Activity",          # дейност/ситуация
    "Stakeholders",      # кои роли участват
    "ExpectedActions",   # какво очакваме от тях
    "NextSteps",         # следващи стъпки/дейности
    "Risks",             # рискове (ако имаш)
    "Mitigations",       # смекчавания (ако имаш)
]

def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    # не гръмваме ако липсва колона; просто я добавяме празна
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""
    return df

def top_matches(df: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Много прост семантичен филтър: търсим по ключови думи в всички важни колони.
    (ако желаеш по-точно – може да се добави embeddings по-късно)
    """
    q = query.lower()
    cols = [c for c in REQUIRED_COLS if c in df.columns]
    mask = False
    for c in cols:
        mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
    hits = df[mask]
    if hits.empty:
        # ако няма директно съвпадение – върни няколко реда за безопасност
        return df.head(top_n)
    return hits.head(top_n)

def compress_rows(rows: pd.DataFrame) -> List[Dict[str, str]]:
    """Обръщаме редовете в компактни записи за подсказката към LLM."""
    items = []
    for _, r in rows.iterrows():
        item = {c: str(r.get(c, "")) for c in REQUIRED_COLS}
        items.append(item)
    return items

# -------------------------------------------------
# Системна подсказка: "не измисляй нови правила"
# -------------------------------------------------
def system_prompt(lang: str, rules_blob: str) -> str:
    bg = (lang == "bg")
    base = (
        "You are BAIT Advisor. Use ONLY the provided table rules to answer. "
        "Do not invent stakeholders, steps, or policies that are not in the table. "
        "Write naturally as a human, concise and clear. If the table doesn't have enough info, ask a short clarifying question."
        if not bg else
        "Ти си BAIT Advisor. Използвай САМО предоставените правила от таблицата. "
        "Не измисляй нови роли, стъпки или политики извън таблицата. "
        "Пиши човешки, кратко и ясно. Ако липсва информация в таблицата – задай кратък уточняващ въпрос."
    )

    header = "TABLE RULES:\n" if not bg else "ПРАВИЛА ОТ ТАБЛИЦАТА:\n"
    return base + "\n\n" + header + rules_blob

def render_rules_blob(items: List[Dict[str, str]], lang: str) -> str:
    """Правим кратък текст от редовете (за да не надхвърляме токени)."""
    lines = []
    for i, it in enumerate(items, 1):
        if lang == "bg":
            lines.append(
                f"{i}) Дейност: {it['Activity']}\n"
                f"   Участници: {it['Stakeholders']}\n"
                f"   Очаквани действия: {it['ExpectedActions']}\n"
                f"   Следващи стъпки: {it['NextSteps']}\n"
                f"   Рискове: {it['Risks']}\n"
                f"   Мерки: {it['Mitigations']}\n"
            )
        else:
            lines.append(
                f"{i}) Activity: {it['Activity']}\n"
                f"   Stakeholders: {it['Stakeholders']}\n"
                f"   Expected actions: {it['ExpectedActions']}\n"
                f"   Next steps: {it['NextSteps']}\n"
                f"   Risks: {it['Risks']}\n"
                f"   Mitigations: {it['Mitigations']}\n"
            )
    blob = "\n".join(lines)
    # ограничаваме подсказката
    if len(blob) > 3000:
        blob = blob[:3000] + " …"
    return blob

# -------------------------------------------------
# Главна функция
# -------------------------------------------------
def interpret_query(
    user_input: str,
    history: List[Dict[str, str]],
    lang: str,
    data_path: str,
) -> str:
    # 1) зареди таблицата и намери най-близките редове
    df = load_table(data_path)
    matches = top_matches(df, user_input, top_n=5)
    rules = compress_rows(matches)
    rules_blob = render_rules_blob(rules, lang)

    # 2) оформяме съобщенията към модела
    sys = {"role": "system", "content": system_prompt(lang, rules_blob)}

    # историята предаваме директно (ако е налична)
    msgs: List[Dict[str, str]] = [sys]
    for m in history[-8:]:  # ограничаваме
        msgs.append(m)

    # текущ въпрос – ще изискаме език по входа
    msgs.append({"role": "user", "content": user_input})

    # 3) извикване на модела
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.2,    # по-детерминирано
            max_tokens=700,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        # връщаме четима грешка към UI
        return f"LLM error: {e}"