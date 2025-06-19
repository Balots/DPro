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
    @decorator
    def run(self) -> pd.DataFrame:
        """
        Обработка пропущенных значений в числовых и категориальных колонках
        с использованием выбранных стратегий.
        """
        df = self.data.copy()

        # Получаем списки колонок по типу
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        # === Числовые колонки ===
        if self.numeric_strategy in ['knn', 'iterative']:
            # Применяем стратегию ко всем числовым колонкам сразу
            if any(df[col].isnull().any() for col in numeric_cols):
                if self.numeric_strategy == 'knn':
                    #logging.info("Применяется KNNImputer")
                    imputer = KNNImputer(n_neighbors=self.knn_k)
                else:  # iterative
                    print('+')
                    #logging.info("Применяется IterativeImputer")
                    imputer = IterativeImputer(max_iter=10, random_state=42)

                try:
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                except Exception as e:
                    print('+')
                    #logging.error(f"Ошибка при применении {self.numeric_strategy}: {e}")
        else:
            # Обрабатываем каждый числовой столбец отдельно
            for col in numeric_cols:
                if df[col].isnull().any():
                    if self.numeric_strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif self.numeric_strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif self.numeric_strategy == 'constant':
                        if col in self.fill_value:
                            df[col] = df[col].fillna(self.fill_value[col])
                        else:
                            raise ValueError(f"Не задано значение для strategy='constant' в столбце '{col}'")
                    else:
                        raise ValueError(f"Неизвестная числовая стратегия: {self.numeric_strategy}")

        # === Категориальные колонки ===
        for col in categorical_cols:
            if df[col].isnull().any():
                if self.categorical_strategy == 'mode':
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        print('+')
                        #logging.warning(f"Невозможно определить моду для столбца '{col}'")
                elif col in self.fill_value:
                    df[col] = df[col].fillna(self.fill_value[col])
                else:
                    #logging.warning(f"Нет fill_value для '{col}', используется 'Unknown'")
                    df[col] = df[col].fillna('Unknown')

        self.result = df
        return self.result
    #@decorator
    def get_answ(self) -> pd.DataFrame:
        """
        Возвращает результат обработки, если необходимо — сначала запускает run().
        """
        if self.result is None:
            self.run()
        return self.result
    #@decorator
    def info(self) -> str:
        """
        Описание используемых стратегий обработки пропущенных значений.
        """
        return (f"Обработка пропущенных: числовые — {self.numeric_strategy}, "
                f"категориальные — {self.categorical_strategy}")


