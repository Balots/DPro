from .base import DataProcessing
from Logger import *

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class HandleMissingValues:
    def __init__(self, data: pd.DataFrame,
                 numeric_strategy: str = 'knn',
                 categorical_strategy: str = 'mode',
                 fill_value: dict = None):
        self.data = data
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value or {}
        self.result = None

    def run(self) -> pd.DataFrame:
        df = self.data.copy()

        # Обрабатываем числовые колонки
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if self.numeric_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif self.numeric_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif self.numeric_strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                elif self.numeric_strategy == 'iterative':
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Обрабатываем категориальные колонки
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if self.categorical_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif col in self.fill_value:
                    df[col] = df[col].fillna(self.fill_value[col])
                else:
                    df[col] = df[col].fillna('Unknown')

        self.result = df
        return self.result
    def get_answ(self):
        if self.result is None:
            self.run()
        return self.result


