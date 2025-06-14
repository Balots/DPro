import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Union
from .base import DataProcessing
from .io.loader import decorator

class NormalizeData(DataProcessing):
    def __init__(self, data: Union[pd.DataFrame, str], 
                 columns: Optional[List] = None, 
                 feature_range: tuple = (0, 1),
                 file_type: Optional[str] = None):
        super().__init__(data, file_type)
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
        return f"Normalizing columns {self.columns} to range {self.feature_range}"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result

class StandardizeData(DataProcessing):
    def __init__(self, data: Union[pd.DataFrame, str], 
                 columns: Optional[List] = None,
                 file_type: Optional[str] = None):
        super().__init__(data, file_type)
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
        return f"Standardizing columns {self.columns} (mean=0, std=1)"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
