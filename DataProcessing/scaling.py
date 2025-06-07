import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .base import DataProcessing
from Logger import *

class NormalizeData(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None, feature_range: tuple = (0, 1)):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.feature_range = feature_range

    @decorator
    def run(self) -> pd.DataFrame:
        df = self.data.copy()
        scaler = MinMaxScaler(feature_range=self.feature_range)
        df[self.columns] = scaler.fit_transform(df[self.columns])
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        return f"Нормализация столбцов {self.columns} с диапазоном {self.feature_range}"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result

class StandardizeData(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()

    @decorator
    def run(self) -> pd.DataFrame:
        df = self.data.copy()
        scaler = StandardScaler()
        df[self.columns] = scaler.fit_transform(df[self.columns])
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        return f"Стандартизация столбцов {self.columns}: приведение к среднему 0 и стандартному отклонению 1"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
