import pandas as pd
import numpy as np
from .base import DataProcessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

class DetectAndRemoveOutliers(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None,
                 method: str = 'iqr', factor: float = 1.5, 
                 contamination: float = 0.05, hampel_threshold: float = 3.0,
                 skewness_threshold: float = 2.0, kurtosis_threshold: float = 3.5):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method.lower()
        self.factor = factor
        self.contamination = contamination
        self.hampel_threshold = hampel_threshold
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.result = None

    def run(self) -> pd.DataFrame:
        df = self.data.copy()
        
        if self.method == 'iqr':
            for col in self.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif self.method == 'hampel':
            for col in self.columns:
                median = df[col].median()
                mad = stats.median_abs_deviation(df[col], scale='normal')
                modified_z = 0.6745 * (df[col] - median) / mad
                df = df[np.abs(modified_z) <= self.hampel_threshold]

        elif self.method == 'percentile':
            for col in self.columns:
                lower_bound = df[col].quantile(0.05)  # 5-й процентиль
                upper_bound = df[col].quantile(0.95)  # 95-й процентиль
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif self.method == 'skewness':
            for col in self.columns:
                skewness = df[col].skew()
                if abs(skewness) > self.skewness_threshold:
                    mean = df[col].mean()
                    std = df[col].std()
                    if skewness > 0:  # Правосторонняя асимметрия
                        df = df[df[col] <= mean + 2 * std]
                    else:  # Левосторонняя асимметрия
                        df = df[df[col] >= mean - 2 * std]

        elif self.method == 'kurtosis':
            for col in self.columns:
                kurtosis = df[col].kurtosis()
                if kurtosis > self.kurtosis_threshold:
                    median = df[col].median()
                    mad = stats.median_abs_deviation(df[col], scale='normal')
                    df = df[np.abs((df[col] - median) / mad) <= 3.5]

        elif self.method == 'zscore':
            z_scores = stats.zscore(df[self.columns])
            mask = (abs(z_scores) < self.factor).all(axis=1)
            df = df[mask]

        elif self.method == 'isolation_forest':
            model = IsolationForest(contamination=self.contamination, random_state=42)
            preds = model.fit_predict(df[self.columns])
            df = df[preds == 1]

        elif self.method == 'lof':
            model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
            preds = model.fit_predict(df[self.columns])
            df = df[preds == 1]

        else:
            raise ValueError(f"Метод '{self.method}' не поддерживается. Доступные методы: "
                             f"IQR, Hampel, Percentile, Skewness, Kurtosis, ZScore, IsolationForest, LOF")

        df = df.reset_index(drop=True)
        self.result = df
        return self.result

    def info(self) -> str:
        info_msg = f"Удаление выбросов методом {self.method.upper()}"
        if self.method in ['iqr', 'zscore']:
            info_msg += f" (factor={self.factor})"
        elif self.method == 'hampel':
            info_msg += f" (threshold={self.hampel_threshold})"
        elif self.method in ['skewness', 'kurtosis']:
            info_msg += f" (threshold={self.skewness_threshold if self.method == 'skewness' else self.kurtosis_threshold})"
        elif self.method in ['isolation_forest', 'lof']:
            info_msg += f" (contamination={self.contamination})"
        info_msg += f" для столбцов {self.columns}"
        return info_msg

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result