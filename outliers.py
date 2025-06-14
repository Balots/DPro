import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union
from .base import DataProcessing
from .io.loader import decorator

class DetectAndRemoveOutliers(DataProcessing):
    def __init__(self, data: Union[pd.DataFrame, str], 
                 columns: Optional[List] = None, 
                 method: str = 'IQR', 
                 factor: float = 1.5, 
                 hampel_threshold: float = 3.5, 
                 skewness_threshold: float = 1.0, 
                 kurtosis_threshold: float = 3.5,
                 file_type: Optional[str] = None):
        super().__init__(data, file_type)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method
        self.factor = factor
        self.hampel_threshold = hampel_threshold
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold

    @decorator
    def run(self) -> pd.DataFrame:
        if self.method == 'IQR':
            return self._run_iqr()
        elif self.method == 'hampel':
            return self._run_hampel()
        elif self.method == 'percentile':
            return self._run_percentile()
        elif self.method == 'skewness':
            return self._run_skewness()
        elif self.method == 'kurtosis':
            return self._run_kurtosis()
        else:
            raise ValueError(f"Outlier detection method '{self.method}' not supported")

    def _run_iqr(self) -> pd.DataFrame:
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

    # Остальные методы (_run_hampel, _run_percentile и т.д.) остаются без изменений
    # ...

    @decorator
    def info(self) -> str:
        info_msg = f"Outlier removal using {self.method} method for columns {self.columns}\n"
        if self.method == 'IQR':
            info_msg += f"Factor: {self.factor}"
        elif self.method == 'hampel':
            info_msg += f"Threshold: {self.hampel_threshold}"
        elif self.method == 'skewness':
            info_msg += f"Skewness threshold: {self.skewness_threshold}"
        elif self.method == 'kurtosis':
            info_msg += f"Kurtosis threshold: {self.kurtosis_threshold}"
        return info_msg

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
