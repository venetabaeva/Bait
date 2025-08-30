import os
import json
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from app.query_engine import BAUniversalQueryEngine

load_dotenv()

# път към master таблицата (остави както е при теб)
DATA_PATH = os.path.join(os.path.dirname(_file_), "data", "master_table.csv")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = BAUniversalQueryEngine(DATA_PATH)

def _detect_lang(text: str) -> str:
    """много проста детекция – ако има кирилица, връща 'bg', иначе 'en'."""
    return "bg" if re.search(r"[А-Яа-я]", text) else "en"

def interpret_query(user_input: str, history: List[Dict[str, str]]) -> str:
    """
    Връща човешки отговор, като:
      - интерпретира user_input през логиката от таблицата
      - отговаря на езика на потребителя
      - ползва кратка история от предишни реплики
    """
    lang = _detect_lang(user_input)

    # 1) извлечи фактори/колони от master таблицата
    factors = engine.get_all_factors()   # напр. ['Activity','Persona','Condition', ...]

    # 2) дай на модела да “разбере” към кои фактори реферира user_input
    sys_map = (
        "You are a classification helper. Map the user's request into a JSON object "
        "that uses keys ONLY from this list of factors (ignore unknown): "
        f"{factors}. Values must be short strings. "
        "If a factor is not present in the user's intent, omit it. "
        "Return ONLY valid JSON, no prose."
    )
    map_messages = [{"role": "system", "content": sys_map},
                    {"role": "user", "content": user_input}]
    mapped = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=map_messages,
        temperature=0.2,
        max_tokens=200,
    ).choices[0].message.content.strip()
    try:
        filters = json.loads(mapped) if mapped else {}
        if not isinstance(filters, dict):
            filters = {}
    except Exception:
        filters = {}

    # 3) филтрирай таблицата с извлечените фактори
    rows = engine.query(**filters) if filters else engine.query()

    # 4) Подготви системен prompt → твърдо изисква отговор на езика на потребителя
    if lang == "bg":
        sys_answer = (
            "Ти си бизнес консултант. Отговаряй САМО на български. "
            "Използвай резултатите от master таблицата като източник на истина. "
            "Не измисляй нови факти извън таблицата; можеш да формулираш човешки тон, "
            "да структурираш в стъпки, но съдържателно се придържай към данните. "
            "Ако таблицата е оскъдна, кажи какво допълнително е нужно."
        )
    else:
        sys_answer = (
            "You are a business advisor. Answer ONLY in English. "
            "Use the master table results as the source of truth. "
            "Do not invent facts beyond the table; you may phrase things naturally, "
            "summarize and structure steps, but content should come from the table. "
            "If the table is insufficient, ask for the missing details."
        )

    # 5) състави контекст: кратка история + извлечени редове от таблицата
    context_blob = engine.render_rows(rows, limit=25)  # текстов извлек от редовете

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_answer}]
    # добави историята (ако има)
    for m in history[-6:]:
        # пазим само ключовото, за да не прехвърлим токени
        messages.append({"role": m["role"], "content": m["content"][:1500]})

    # добавяме контекст от таблицата и последния потребителски въпрос
    messages.append({
        "role": "system",
        "content": f"Master table context (rows):\n{context_blob}"
    })
    messages.append({"role": "user", "content": user_input})

    # 6) окончателен отговор
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    ).choices[0].message.content.strip()

    return answer