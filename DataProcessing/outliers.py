import pandas as pd
import numpy as np
from .base import DataProcessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, zscore

class DetectAndRemoveOutliers(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None, method: str = 'IQR', factor: float = 1.5, contamination: float = 0.05):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method
        self.factor = factor
        self.contamination = contamination
        self.result = None

    def _select_numeric_columns(self):
        return self.data.select_dtypes(include='number').columns.tolist()

    def _iqr_method(self, df):
        for col in self.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def _zscore_method(self, df):
        z_scores = zscore(df[self.columns])
        mask = (np.abs(z_scores) < 3).all(axis=1)
        return df[mask]

    def _isolation_forest_method(self, df):
        model = IsolationForest(contamination=self.contamination, random_state=42)
        preds = model.fit_predict(df[self.columns])
        return df[preds == 1]

    def _lof_method(self, df):
        model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        preds = model.fit_predict(df[self.columns])
        return df[preds == 1]

    def _evaluate_skewness(self, df):
        if df.empty:
            return np.inf
        return np.mean([abs(skew(df[col].dropna())) for col in self.columns])

    def run(self) -> pd.DataFrame:
        df = self.data.copy()

        if self.method.lower() == 'iqr':
            df_filtered = self._iqr_method(df)
        elif self.method.lower() == 'zscore':
            df_filtered = self._zscore_method(df)
        elif self.method.lower() == 'isolation_forest':
            df_filtered = self._isolation_forest_method(df)
        elif self.method.lower() == 'lof':
            df_filtered = self._lof_method(df)
        elif self.method.lower() == 'auto':
            methods = {
                'IQR': self._iqr_method,
                'Z-Score': self._zscore_method,
                'IsolationForest': self._isolation_forest_method,
                'LOF': self._lof_method,
            }
            best_score = np.inf
            best_method = None
            best_df = None
            for name, method_func in methods.items():
                temp_df = method_func(df.copy())
                score = self._evaluate_skewness(temp_df)
                # Логируем результаты
                print(f"Метод {name}: среднее |skew| = {score:.4f}, оставлено строк: {temp_df.shape[0]}")
                if score < best_score:
                    best_score = score
                    best_method = name
                    best_df = temp_df
            print(f"Выбран лучший метод: {best_method} с |skew| = {best_score:.4f}")
            df_filtered = best_df
        else:
            raise ValueError(f"Метод '{self.method}' не поддерживается")

        self.result = df_filtered.reset_index(drop=True)
        return self.result

    def info(self) -> str:
        return f"Удаление выбросов методом {self.method} с фактором {self.factor} для столбцов {self.columns}"

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
