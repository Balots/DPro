import pandas as pd
import numpy as np
from .base import DataProcessing
from Logger import *

class DetectAndRemoveOutliers(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None, method: str = 'IQR', 
                 factor: float = 1.5, hampel_threshold: float = 3.5, 
                 skewness_threshold: float = 1.0, kurtosis_threshold: float = 3.5):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method
        self.factor = factor
        self.hampel_threshold = hampel_threshold
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold

    @decorator
    def run(self) -> pd.DataFrame:
        if self.method == 'IQR':
            return self.run_iqr()
        elif self.method == 'hampel':
            return self.run_hampel()
        elif self.method == 'percentile':
            return self.run_percentile()
        elif self.method == 'skewness':
            return self.run_skewness()
        elif self.method == 'kurtosis':
            return self.run_kurtosis()
        else:
            raise ValueError(f"Метод обнаружения выбросов '{self.method}' не поддерживается")

    @decorator
    def run_iqr(self) -> pd.DataFrame:
        """Метод межквартильного размаха (IQR)"""
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

    @decorator
    def run_hampel(self) -> pd.DataFrame:
        """Модифицированный Z-score (фильтр Хемпеля)"""
        df = self.data.copy()
        for col in self.columns:
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            if mad != 0:
                modified_z = 0.6745 * (df[col] - median) / mad
                df = df[np.abs(modified_z) <= self.hampel_threshold]
        df = df.reset_index(drop=True)
        self.result = df
        return self.result

    @decorator
    def run_percentile(self) -> pd.DataFrame:
        """Метод процентилей (P5-P95)"""
        df = self.data.copy()
        for col in self.columns:
            p5 = df[col].quantile(0.05)
            p95 = df[col].quantile(0.95)
            df = df[(df[col] >= p5) & (df[col] <= p95)]
        df = df.reset_index(drop=True)
        self.result = df
        return self.result

    @decorator
    def run_skewness(self) -> pd.DataFrame:
        """Метод на основе коэффициента асимметрии"""
        df = self.data.copy()
        for col in self.columns:
            col_data = df[col]
            mean = col_data.mean()
            std = col_data.std()
            skewness = col_data.skew()
            
            if abs(skewness) > self.skewness_threshold:
                if skewness > 0:
                    df = df[col_data <= mean + 2 * std]
                else:
                    df = df[col_data >= mean - 2 * std]
        df = df.reset_index(drop=True)
        self.result = df
        return self.result

    @decorator
    def run_kurtosis(self) -> pd.DataFrame:
        """Метод на основе эксцесса"""
        df = self.data.copy()
        for col in self.columns:
            col_data = df[col]
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            kurtosis = col_data.kurtosis()
            
            if kurtosis > self.kurtosis_threshold and mad != 0:
                normalized = np.abs((col_data - median) / mad)
                df = df[normalized <= 3.5]
        df = df.reset_index(drop=True)
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        info_msg = f"Удаление выбросов методом {self.method} для столбцов {self.columns}\n"
        if self.method == 'IQR':
            info_msg += f"Фактор: {self.factor}"
        elif self.method == 'hampel':
            info_msg += f"Порог: {self.hampel_threshold}"
        elif self.method == 'skewness':
            info_msg += f"Порог асимметрии: {self.skewness_threshold}"
        elif self.method == 'kurtosis':
            info_msg += f"Порог эксцесса: {self.kurtosis_threshold}"
        return info_msg

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result