import pandas as pd
from .base import DataProcessing
from Logger import *


class HandleMissingValues(DataProcessing):
    def __init__(self, data: pd.DataFrame, numeric_strategy: str = 'mean',
                 categorical_strategy: str = 'mode', fill_value: dict = None, columns: list = None):
        super().__init__(data)
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value or {}
        self.columns = columns or self.data.columns.tolist()

    @decorator
    def run(self) -> pd.DataFrame:
        df = self.data.copy()
        for col in self.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = self.numeric_strategy
                else:
                    strategy = self.categorical_strategy

                if strategy in ['mean', 'median'] and not pd.api.types.is_numeric_dtype(df[col]):
                    logging.warning(f"Стратегия '{strategy}' неприменима к нечисловому столбцу '{col}'")
                    continue

                if strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mode':
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        logging.warning(f"Невозможно вычислить моду для столбца '{col}'")
                elif strategy == 'constant':
                    if col in self.fill_value:
                        df[col] = df[col].fillna(self.fill_value[col])
                    else:
                        raise ValueError(f"Для столбца '{col}' не задано значение для стратегии 'constant'")
                else:
                    raise ValueError(f"Неизвестная стратегия заполнения: {strategy}")
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        return f"Обработка пропущенных значений: числовые - {self.numeric_strategy}, категориальные - {self.categorical_strategy}"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result
