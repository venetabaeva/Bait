# app/query_engine.py
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

class BAUniversalQueryEngine:
    """
    Лек „двигател“ за заявки към master таблицата.
    - Зарежда CSV от подадения път.
    - Позволява филтър по произволни колони.
    - Има удобен текстов рендер на редовете (render_rows).
    """

    def __init__(self, master_table_path: str | Path):
        # Разрешаваме относителни пътища спрямо този файл
        base = Path(_file_).resolve().parent
        p = Path(master_table_path)
        if not p.is_absolute():
            p = (base / p).resolve()

        # Чети CSV (опит с UTF-8, после с utf-8-sig)
        try:
            df = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="utf-8-sig")

        # Запази оригинала
        self.df_original = df.copy()

        # Подготви „почистена“ версия за лесно филтриране
        df = df.fillna("")
        # Strip за всички string клетки
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()

        self.df = df
        self.path = p

    # --------------------------
    # Инфо за колоните (факторите)
    # --------------------------
    def get_all_factors(self) -> List[str]:
        """Списък с наличните колони (фактори) в таблицата."""
        return [str(c) for c in self.df.columns]

    # --------------------------
    # Основна заявка
    # --------------------------
    def query(
        self,
        contains: bool = True,
        case_insensitive: bool = True,
        limit: Optional[int] = None,
        **filters: Any,
    ) -> pd.DataFrame:
        """
        Филтрирай по всякаква комбинация от колони:
        query(Activity="Plan Business Analysis", Persona="Sponsor")

        Параметри:
        - contains: ако True → използва substring match; ако False → точно съвпадение
        - case_insensitive: ако True → прави сравнения нечувствителни към регистър
        - limit: ако е подадено → връща първите N реда
        - **filters: key=column, value=търсена стойност (празни/None се игнорират)
        """
        if not filters:
            result = self.df.copy()
        else:
            result = self.df.copy()
            for key, value in filters.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    continue
                if key not in result.columns:
                    # Ако колоната не съществува, прескачаме я „тихо“
                    continue

                series = result[key].astype(str)
                val = str(value)

                if case_insensitive:
                    series = series.str.lower()
                    val = val.lower()

                if contains:
                    mask = series.str.contains(val, na=False)
                else:
                    mask = (series == val)

                result = result[mask]

                # Рано излизане, ако вече е празно
                if result.empty:
                    break

        if limit is not None and limit > 0:
            result = result.head(limit)

        return result.reset_index(drop=True)

    # --------------------------
    # Свободен „full-text“ search през всички колони
    # --------------------------
    def search_any(
        self,
        text: str,
        case_insensitive: bool = True,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Търси текст във ВСИЧКИ колони, ползвайки substring match.
        Полезно когато искаш бърза проверка с едно изречение/дума.
        """
        if not text or not str(text).strip():
            return self.df.head(0)  # празен резултат със същите колони

        val = str(text)
        df = self.df.copy()

        mask = None
        for col in df.columns:
            col_series = df[col].astype(str)
            if case_insensitive:
                col_series = col_series.str.lower()
                cur = val.lower()
            else:
                cur = val
            m = col_series.str.contains(cur, na=False)
            mask = m if mask is None else (mask | m)

        result = df[mask] if mask is not None else df.head(0)
        if limit is not None and limit > 0:
            result = result.head(limit)
        return result.reset_index(drop=True)

    # --------------------------
    # Представяне на редове като текст за „човешки“ рендер
    # --------------------------
    def render_rows(
        self,
        df: Optional[pd.DataFrame] = None,
        limit: int = 25,
        columns: Optional[Iterable[str]] = None,
        bullet: str = "• ",
        sep: str = "  |  ",
        skip_empty: bool = True,
    ) -> str:
        """
        Превръща редовете в четим текст:
        • Col1: Val1  |  Col2: Val2
        • Col1: Val1  |  Col2: Val2
        ...
        """
        if df is None:
            df = self.df

        if df is None or df.empty:
            return "No rows."

        small = df.head(limit).copy()

        # Избери кои колони да показваш (по подразбиране — всички)
        if columns:
            use_cols = [c for c in columns if c in small.columns]
            if not use_cols:
                use_cols = list(small.columns)
        else:
            use_cols = list(small.columns)

        lines: List[str] = []
        for _, row in small.iterrows():
            parts: List[str] = []
            for c in use_cols:
                v = row[c]
                s = "" if pd.isna(v) else str(v).strip()
                if skip_empty and not s:
                    continue
                parts.append(f"{c}: {s}")
            if not parts:
                continue
            lines.append(bullet + sep.join(parts))

        return "\n".join(lines) if lines else "No rows."

    # --------------------------
    # Удобен преглед
    # --------------------------
    def preview(self, limit: int = 5) -> str:
        """Кратък преглед на началните редове като текст."""
        return self.render_rows(self.df.head(limit), limit=limit)

    # --------------------------
    # Конвертиране към list[dict] (ако ти потрябва за JSON)
    # --------------------------
    def to_dict_rows(self, df: Optional[pd.DataFrame] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if df is None:
            df = self.df
        if limit is not None and limit > 0:
            df = df.head(limit)
        return df.to_dict(orient="records")

_all_ = ["BAUniversalQueryEngine"]