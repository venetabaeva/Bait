import pandas as pd

class BAUniversalQueryEngine:
    def __init__(self, master_table_path: str):
        self.df = pd.read_csv(master_table_path).fillna("")

    def get_all_factors(self):
        # фактори = имената на колоните
        return list(self.df.columns)

    def query(self, **kwargs):
        """
        Филтрира таблицата по подадени фактори (case-insensitive).
        Пример: query(Activity="Plan Business Analysis", Persona="Sponsor")
        """
        result = self.df.copy()
        for key, val in kwargs.items():
            if key in result.columns and val:
                result = result[result[key].astype(str).str.lower() == str(val).lower()]
        return result

    def summarize_rows(self, rows: pd.DataFrame) -> str:
        """
        Прави човешко резюме на подбраните редове.
        Опитва се да използва често срещани колони; ако ги няма — събира стойности по редове.
        """
        if rows.empty:
            return "No exact rule match in the master table. I’ll still reason based on the closest factors."

        parts = []

        # Първо — ако има колони с указания/действия
        for col in ["Recommended actions", "Actions", "Advice", "Guidance"]:
            if col in rows.columns:
                uniq = [x for x in rows[col].astype(str).unique() if x]
                if uniq:
                    parts.append("Recommended actions:\n- " + "\n- ".join(uniq))
                    break

        # Добавим контекст, ако има
        for col in ["Rationale", "Outcome", "Notes", "Considerations"]:
            if col in rows.columns:
                uniq = [x for x in rows[col].astype(str).unique() if x]
                if uniq:
                    parts.append(f"{col}:\n- " + "\n- ".join(uniq))

        if not parts:
            # Generic fallback: изредете уникални стойности по няколко ключови колони
            sample_cols = rows.columns[:6]
            lines = []
            for _, r in rows.iterrows():
                items = [f"{c}: {str(r[c]).strip()}" for c in sample_cols if str(r[c]).strip()]
                if items:
                    lines.append("• " + " | ".join(items))
            if lines:
                parts.append("\n".join(lines))

        return "\n\n".join(parts)