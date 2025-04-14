import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class DataProcessing(ABC):
    def __init__(self, data: pd.DataFrame):
        """
        Инициализация с копией переданного DataFrame
        """
        self.data = data.copy()
        self.result = None

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Выполнение обработки данных
        """
        pass

    @abstractmethod
    def info(self) -> str:
        """
        Описание выполняемой обработки
        """
        pass

    @abstractmethod
    def get_answ(self) -> pd.DataFrame:
        """
        Возвращает результат обработки
        """
        pass

    def _select_numeric_columns(self) -> list:
        """
        Вспомогательный метод для выбора числовых столбцов
        """
        return self.data.select_dtypes(include=np.number).columns.tolist()
