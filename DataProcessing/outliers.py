import pandas as pd
from .base import DataProcessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

class DetectAndRemoveOutliers(DataProcessing):
    def __init__(self, data: pd.DataFrame, columns: list = None,
                 method: str = 'isolation_forest', factor: float = 1.5, contamination: float = 0.05):
        super().__init__(data)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method
        self.factor = factor
        self.contamination = contamination
        self.result = None

    def run(self) -> pd.DataFrame:
        df = self.data.copy()
        if self.method == 'IQR':
            for col in self.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

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
            raise ValueError(f"Метод '{self.method}' не поддерживается")

        df = df.reset_index(drop=True)
        self.result = df
        return self.result


    def info(self) -> str:
        return f"Удаление выбросов методом {self.method} (factor={self.factor}, contamination={self.contamination}) для столбцов {self.columns}"

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result

