import pandas as pd
from .base import DataProcessing
from .io.loader import decorator

class CleanData(DataProcessing):
    @decorator
    def run(self) -> pd.DataFrame:
        """Remove duplicates and reset index"""
        self.result = self.data.drop_duplicates().reset_index(drop=True)
        return self.result

    @decorator
    def info(self) -> str:
        return "Removing duplicates and resetting index"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
