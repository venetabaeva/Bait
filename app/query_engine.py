import pandas as pd
import re

class BAUniversalQueryEngine:
    def __init__(self, master_table_path: str):
        self.df = pd.read_csv(master_table_path).fillna("")
        # normalize convenience columns for robust matching
        self.norm = self.df.copy()
        for c in self.norm.columns:
            if self.norm[c].dtype == object:
                self.norm[c] = self.norm[c].astype(str).str.strip().str.lower()

    def get_all_factors(self):
        # columns available in the table
        return list(self.df.columns)

    def query_contains(self, **kwargs):
        """
        Flexible case-insensitive 'contains' matching on provided columns.
        Example: query_contains(Activity="stakeholder conflict")
        """
        if not kwargs:
            return self.df.head(0)

        mask = None
        for col, val in kwargs.items():
            if col not in self.norm.columns or not isinstance(val, str):
                continue
            v = val.strip().lower()
            if not v:
                continue
            # word-boundary-ish contains (fallback to simple contains)
            pattern = re.escape(v)
            colmask = self.norm[col].str.contains(pattern, na=False, regex=True)
            mask = colmask if mask is None else (mask & colmask)

        if mask is None:
            return self.df.head(0)

        return self.df[mask]