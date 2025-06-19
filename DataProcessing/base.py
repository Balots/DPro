import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional
# Правильный абсолютный путь от корня проекта
#from io.loader import DataLoader

class DataProcessing(ABC):
    def __init__(self, data: Union[pd.DataFrame, str], file_type: Optional[str] = None):
        """
        Базовый класс для обработки данных.
        
        Параметры:
        -----------
        data : Union[pd.DataFrame, str]
            Может быть либо DataFrame, либо путь к файлу с данными
        file_type : Optional[str]
            Тип файла (если data - строка), например 'xlsx', 'json', 'parquet'
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            self.data = DataLoader.load_data(data, file_type)
        else:
            raise ValueError("Данные должны быть либо DataFrame, либо путь к файлу")
            
        self.result = None

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Выполнить обработку данных"""
        pass

    @abstractmethod
    def info(self) -> str:
        """Получить описание выполняемой обработки"""
        pass

    @abstractmethod
    def get_answ(self) -> pd.DataFrame:
        """Получить результат обработки"""
        pass

    def _select_numeric_columns(self) -> list:
        """
        Выбрать числовые столбцы
        
        Возвращает:
        -----------
        list
            Список числовых столбцов
        """
        return self.data.select_dtypes(include='number').columns.tolist()

    def save_result(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> None:
        """
        Сохранить результат обработки в файл
        
        Параметры:
        -----------
        file_path : str
            Путь для сохранения файла
        file_type : Optional[str]
            Тип файла (если не указан, определяется из расширения)
        **kwargs : dict
            Дополнительные параметры для сохранения
        """
        if self.result is None:
            self.run()
        DataLoader.save_data(self.result, file_path, file_type, **kwargs)
