import os
import json
from typing import List, Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# зареди .env (ако има локално)
load_dotenv()

# Модел – можеш да смениш при нужда
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --------- Помощни функции върху таблицата ---------

def load_table(data_path: str) -> pd.DataFrame:
    """
    Чете master CSV и нормализира празните клетки.
    """
    if not os.path.isabs(data_path):
        # направи пътя относителен спрямо текущия файл
        data_path = os.path.join(os.path.dirname(_file_), data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Master table not found at: {data_path}")

    df = pd.read_csv(data_path)
    # нормализирай празнини и типове
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)
    return df


def _row_text(df: pd.DataFrame, i: int) -> str:
    """
    Сглоби индекс i на df в един ред текст (всички колони).
    """
    row = df.iloc[i]
    # всички колони като "ColName: value"
    parts = []
    for col in df.columns:
        val = str(row[col]).strip()
        if val:
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


def _simple_overlap_score(query: str, text: str) -> int:
    """
    Много проста метрика: брои съвпадения на думи (case-insensitive).
    Не е перфектна, но работи стабилно без външни зависимости.
    """
    q_tokens = {t for t in query.lower().split() if len(t) > 2}
    t_tokens = {t for t in text.lower().split() if len(t) > 2}
    return len(q_tokens & t_tokens)


def find_top_matches(user_input: str, df: pd.DataFrame, top_n: int = 5) -> List[Tuple[int, int]]:
    """
    Връща списък от (index, score) за top_n най-близки редове.
    """
    scores: List[Tuple[int, int]] = []
    for i in range(len(df)):
        txt = _row_text(df, i)
        score = _simple_overlap_score(user_input, txt)
        if score > 0:
            scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def compress_rows(df: pd.DataFrame, matches: List[Tuple[int, int]], lang: str) -> str:
    """
    Прави кратък „контекст“ от най-подходящите редове, в удобен за LLM вид.
    """
    if not matches:
        return ""

    lines = []
    for idx, sc in matches:
        row = df.iloc[idx].to_dict()
        # По-четим формат
        block = []
        for k, v in row.items():
            v = (v or "").strip()
            if v:
                block.append(f"- {k}: {v}")
        lines.append("\n".join(block))

    blob = "\n\n---\n\n".join(lines)
    return blob


def system_prompt(lang: str, rules_blob: str) -> str:
    """
    Инструкции към модела – отговаряй само по предоставените правила.
    """
    # език: "bg" или "en" – не е критично; използваме като hint
    lang_hint = "Bulgarian" if (lang or "").lower().startswith("bg") else "English"

    return f"""
You are BAIT Advisor.
Answer strictly based ONLY on the factual content under 'RULES'.
Do NOT invent stakeholders, actions, or details that are not present in the rules.
Write a fluent, natural, human-style answer in {lang_hint}.
If the rules do not contain enough info, say briefly what is missing and ask 1–2 clarifying questions.

RULES (tabular facts, already extracted):
{rules_blob}
""".strip()


# --------- Главна функция, която вика LLM ---------

def interpret_query(
    user_input: str,
    history: List[Dict[str, str]] | None = None,
    lang: str | None = "bg",
    data_path: str | None = None,
) -> str:
    """
    1) Зарежда таблицата.
    2) Намира най-близките редове.
    3) Сглобява „правила“ за LLM.
    4) Моли модела да формулира човешки отговор без да измисля факти.
    """
    # 1) път към master_table.csv (относително към файла)
    if data_path is None:
        data_path = os.path.join(os.path.dirname(_file_), "data", "master_table.csv")
    elif not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(_file_), data_path)

    df = load_table(data_path)

    # 2) топ съвпадения
    matches = find_top_matches(user_input, df, top_n=5)
    rules_blob = compress_rows(df, matches, lang)

    # 3) системен prompt
    sys_txt = system_prompt(lang or "bg", rules_blob)

    # 4) история (ако има) + текущ потребителски вход
    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_txt}]
    if history:
        # очакваме елементи {"role": "user"/"assistant", "content": "..."}
        msgs.extend(history[-8:])  # малък контекст
    msgs.append({"role": "user", "content": user_input})

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.2,   # по-детерминистично
            max_tokens=700,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        # да върнем кратко, четимо съобщение към UI
        return f"LLM error: {e}"