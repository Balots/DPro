import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union
from .base import DataProcessing
from .io.loader import decorator

class DetectAndRemoveOutliers(DataProcessing):
    """
    Класс для обнаружения и удаления выбросов в данных.
    
    Поддерживает несколько методов обнаружения выбросов:
    - IQR (межквартильный размах)
    - Hampel (модифицированный Z-критерий)
    - Percentile (процентили)
    - Skewness (асимметрия)
    - Kurtosis (эксцесс)
    """
    
    def __init__(self, data: Union[pd.DataFrame, str], 
                 columns: Optional[List] = None, 
                 method: str = 'IQR', 
                 factor: float = 1.5, 
                 hampel_threshold: float = 3.5, 
                 skewness_threshold: float = 1.0, 
                 kurtosis_threshold: float = 3.5,
                 file_type: Optional[str] = None):
        """
        Инициализация детектора выбросов.
        
        Параметры:
        ----------
        data : Union[pd.DataFrame, str]
            Входные данные (DataFrame или путь к файлу)
        columns : List, optional
            Список колонок для анализа (по умолчанию все числовые колонки)
        method : str, optional
            Метод обнаружения выбросов (по умолчанию 'IQR')
        factor : float, optional
            Множитель для метода IQR (по умолчанию 1.5)
        hampel_threshold : float, optional
            Порог для метода Hampel (по умолчанию 3.5)
        skewness_threshold : float, optional
            Порог асимметрии (по умолчанию 1.0)
        kurtosis_threshold : float, optional
            Порог эксцесса (по умолчанию 3.5)
        file_type : str, optional
            Тип файла, если data - строка
        """
        super().__init__(data, file_type)
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.method = method
        self.factor = factor
        self.hampel_threshold = hampel_threshold
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold

    @decorator
    def run(self) -> pd.DataFrame:
        """
        Запускает обнаружение и удаление выбросов согласно выбранному методу.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с удаленными выбросами
        """
        if self.method == 'IQR':
            return self._run_iqr()
        elif self.method == 'hampel':
            return self._run_hampel()
        elif self.method == 'percentile':
            return self._run_percentile()
        elif self.method == 'skewness':
            return self._run_skewness()
        elif self.method == 'kurtosis':
            return self._run_kurtosis()
        else:
            raise ValueError(f"Метод обнаружения выбросов '{self.method}' не поддерживается")

    def _run_iqr(self) -> pd.DataFrame:
        """
        Метод межквартильного размаха (IQR) для обнаружения выбросов.
        
        Вычисляет границы: 
        - Нижняя: Q1 - factor * IQR
        - Верхняя: Q3 + factor * IQR
        """
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
    def info(self) -> str:
        """
        Возвращает информацию о текущих настройках детектора.
        
        Возвращает:
        -----------
        str
            Описание метода и параметров обнаружения выбросов
        """
        info_msg = (f"Удаление выбросов методом {self.method} "
                   f"для колонок {self.columns}\n")
        if self.method == 'IQR':
            info_msg += f"Множитель: {self.factor}"
        elif self.method == 'hampel':
            info_msg += f"Порог: {self.hampel_threshold}"
        elif self.method == 'skewness':
            info_msg += f"Порог асимметрии: {self.skewness_threshold}"
        elif self.method == 'kurtosis':
            info_msg += f"Порог эксцесса: {self.kurtosis_threshold}"
        return info_msg

    @decorator
    def get_answ(self) -> pd.DataFrame:
        """
        Возвращает результат обработки.
        
        Если обработка еще не выполнялась, запускает ее автоматически.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с удаленными выбросами
        """
        if self.result is None:
            self.run()
        return self.result
