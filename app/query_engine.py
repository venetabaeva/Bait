# app/query_engine.py
import pandas as pd
import re

EVIDENCE_FIELDS = [
    # сложи тук колоните, които са „истините“ за отговор: напр.
    "Activity", "Stakeholder", "Expectation", "Next actions", "Risks", "Notes"
]

class BAUniversalQueryEngine:
    def __init__(self, master_table_path: str):
        self.df = pd.read_csv(master_table_path).fillna("")
        # нормализирани имена на колони
        self.cols = {c.lower().strip(): c for c in self.df.columns}

    def _find_columns(self, text: str):
        """опит за намиране на колони по ключови думи в user input (фъзи/синоними)."""
        t = text.lower()
        hits = []
        for k, orig in self.cols.items():
            tokens = re.split(r"[^a-z0-9]+", k)
            if any(tok and tok in t for tok in tokens):
                hits.append(orig)
        return list(set(hits)) or list(self.df.columns)

    def query(self, **filters):
        """filters идват от LLM (интерпретация). Пример: Activity='Requirements conflict'."""
        if not filters:
            return [], self.df.head(0)  # няма филтри → празно съвпадение

        result = self.df.copy()
        for key, val in filters.items():
            if not val:
                continue
            # позволи фъзи мач по съдържание
            key_real = self.cols.get(key.lower(), key)
            if key_real in result.columns:
                v = str(val).strip().lower()
                result = result[result[key_real].astype(str).str.lower().str.contains(re.escape(v))]
        # намали само до полетата за доказателства
        present_fields = [c for c in EVIDENCE_FIELDS if c in result.columns]
        evidence = result[present_fields].copy() if present_fields else result.copy()
        return present_fields, evidence