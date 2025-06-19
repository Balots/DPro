import pandas as pd
from .base import DataProcessing
from Logger import *

class CleanData(DataProcessing):
    @decorator
    def run(self) -> pd.DataFrame:
        self.result = self.data.drop_duplicates().reset_index(drop=True)
        return self.result

    @decorator
    def info(self) -> str:
        return "Удаление дубликатов и сброс индексов"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
