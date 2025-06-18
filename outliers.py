import pandas as pd
from .base import DataProcessing

class DetectAndRemoveOutliers(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None, method: str = 'IQR', factor: float = 1.5):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method
        self.factor = factor

    def run(self) -> pd.DataFrame:
        if self.method != 'IQR':
            raise ValueError(f"Метод обнаружения выбросов '{self.method}' не поддерживается")

        df = self.data.copy()
        for col in self.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        df = df.reset_index(drop=True)
        self.result = df
        return self.result

    def info(self) -> str:
        return f"Удаление выбросов методом {self.method} с фактором {self.factor} для столбцов {self.columns}"

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
