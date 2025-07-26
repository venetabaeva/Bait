import pandas as pd

class BAUniversalQueryEngine:
    def __init__(self, master_table_path):
        self.df = pd.read_csv(master_table_path)
        self.df = self.df.fillna('')  # Replace NaN with empty string

    def query(self, **kwargs):
        """
        Query the master table with any combination of factors.
        Example: query(Activity="Plan Business Analysis", Persona="Sponsor")
        """
        result = self.df.copy()
        for key, value in kwargs.items():
            if key in self.df.columns and value:
                result = result[result[key].str.lower() == value.lower()]
        return result

    def get_all_factors(self):
        """
        Returns the list of all columns (factors) in the table.
        """
        return list(self.df.columns)
