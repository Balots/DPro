import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from .base import DataProcessing
from .io.loader import decorator

class HandleMissingValues(DataProcessing):
    """
    Класс для обработки пропущенных значений в данных.
    
    Поддерживает различные стратегии обработки для числовых и категориальных данных.
    """
    
    def __init__(self, data: Union[pd.DataFrame, str], 
                 numeric_strategy: str = 'mean',
                 categorical_strategy: str = 'mode', 
                 fill_value: Optional[Dict] = None, 
                 columns: Optional[List] = None,
                 file_type: Optional[str] = None):
        """
        Инициализация обработчика пропущенных значений.
        
        Параметры:
        ----------
        data : Union[pd.DataFrame, str]
            Входные данные (DataFrame или путь к файлу)
        numeric_strategy : str, optional
            Стратегия для числовых данных (по умолчанию 'mean')
        categorical_strategy : str, optional
            Стратегия для категориальных данных (по умолчанию 'mode')
        fill_value : Dict, optional
            Словарь значений для заполнения (для стратегии 'constant')
        columns : List, optional
            Список колонок для обработки (по умолчанию все колонки)
        file_type : str, optional
            Тип файла, если data - строка
        """
        super().__init__(data, file_type)
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value or {}
        self.columns = columns or self.data.columns.tolist()

    @decorator
    def run(self) -> pd.DataFrame:
        """
        Выполняет обработку пропущенных значений согласно заданным стратегиям.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с обработанными пропущенными значениями
        """
        df = self.data.copy()
        for col in self.columns:
            if df[col].isnull().sum() > 0:
                # Определяем стратегию в зависимости от типа данных
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = self.numeric_strategy
                else:
                    strategy = self.categorical_strategy

                # Проверка применимости стратегии
                if strategy in ['mean', 'median'] and not pd.api.types.is_numeric_dtype(df[col]):
                    logging.warning(f"Стратегия '{strategy}' неприменима к нечисловой колонке '{col}'")
                    continue

                # Применяем выбранную стратегию
                if strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mode':
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        logging.warning(f"Не удалось вычислить моду для колонки '{col}'")
                elif strategy == 'constant':
                    if col in self.fill_value:
                        df[col] = df[col].fillna(self.fill_value[col])
                    else:
                        raise ValueError(f"Для колонки '{col}' не указано значение для стратегии 'constant'")
                else:
                    raise ValueError(f"Неизвестная стратегия заполнения: {strategy}")
        
        self.result = df
        return self.result

    @decorator
    def info(self) -> str:
        """
        Возвращает информацию о стратегиях обработки.
        
        Возвращает:
        -----------
        str
            Строка с описанием используемых стратегий
        """
        return (f"Обработка пропущенных значений: числовые - {self.numeric_strategy}, "
                f"категориальные - {self.categorical_strategy}")

    @decorator
    def get_answ(self) -> pd.DataFrame:
        """
        Возвращает результат обработки.
        
        Если обработка еще не выполнялась, запускает ее автоматически.
        
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с обработанными пропущенными значениями
        """
        if self.result is None:
            self.run()
        return self.result
