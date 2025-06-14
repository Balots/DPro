import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Optional, Union
from .base import DataProcessing
from .io.loader import decorator

class NormalizeData(DataProcessing):
    """
    Класс для нормализации данных (приведение к заданному диапазону).
    
    Использует MinMaxScaler для масштабирования признаков к заданному диапазону (по умолчанию [0, 1]).
    """
    
    def __init__(self, data: Union[pd.DataFrame, str], 
                 columns: Optional[List] = None, 
                 feature_range: tuple = (0, 1),
                 file_type: Optional[str] = None):
        """
        Инициализация нормализатора данных.
        
        Параметры:
        ----------
        data : Union[pd.DataFrame, str]
            Входные данные (DataFrame или путь к файлу)
        columns : List, optional
            Список колонок для нормализации (по умолчанию все числовые колонки)
        feature_range : tuple, optional
            Диапазон для нормализации (по умолчанию (0, 1))
        file_type : str, optional
            Тип файла, если data - строка
        """
        super().__init__(data, file_type)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.feature_range = feature_range

    @decorator
    def run(self) -> pd.DataFrame:
        """
        Выполняет нормализацию данных.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с нормализованными значениями в указанных колонках
        """
        df = self.data.copy()
        scaler = MinMaxScaler(feature_range=self.feature_range)
        df[self.columns] = scaler.fit_transform(df[self.columns])
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        """
        Возвращает информацию о параметрах нормализации.
        
        Возвращает:
        -----------
        str
            Строка с описанием параметров нормализации
        """
        return f"Нормализация колонок {self.columns} к диапазону {self.feature_range}"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        """
        Возвращает результат нормализации.
        
        Если нормализация еще не выполнялась, запускает ее автоматически.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с нормализованными данными
        """
        if self.result is None:
            self.run()
        return self.result


class StandardizeData(DataProcessing):
    """
    Класс для стандартизации данных (приведение к среднему=0, std=1).
    
    Использует StandardScaler для преобразования данных.
    """
    
    def __init__(self, data: Union[pd.DataFrame, str], 
                 columns: Optional[List] = None,
                 file_type: Optional[str] = None):
        """
        Инициализация стандартизатора данных.
        
        Параметры:
        ----------
        data : Union[pd.DataFrame, str]
            Входные данные (DataFrame или путь к файлу)
        columns : List, optional
            Список колонок для стандартизации (по умолчанию все числовые колонки)
        file_type : str, optional
            Тип файла, если data - строка
        """
        super().__init__(data, file_type)
        self.columns = columns if columns is not None else self._select_numeric_columns()

    @decorator
    def run(self) -> pd.DataFrame:
        """
        Выполняет стандартизацию данных.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame со стандартизованными значениями (среднее=0, std=1)
        """
        df = self.data.copy()
        scaler = StandardScaler()
        df[self.columns] = scaler.fit_transform(df[self.columns])
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        """
        Возвращает информацию о параметрах стандартизации.
        
        Возвращает:
        -----------
        str
            Строка с описанием параметров стандартизации
        """
        return f"Стандартизация колонок {self.columns} (среднее=0, std=1)"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        """
        Возвращает результат стандартизации.
        
        Если стандартизация еще не выполнялась, запускает ее автоматически.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame со стандартизованными данными
        """
        if self.result is None:
            self.run()
        return self.result
