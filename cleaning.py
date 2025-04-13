import pandas as pd
from .base import DataProcessing

class CleanData(DataProcessing):
    def run(self) -> pd.DataFrame:
        self.result = self.data.drop_duplicates().reset_index(drop=True)
        return self.result

    def info(self) -> str:
        return "Удаление дубликатов и сброс индексов"

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
