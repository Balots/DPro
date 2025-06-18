import pandas as pd
from .base import DataProcessing
from .io.loader import decorator

class CleanData(DataProcessing):
    """
    Класс для очистки данных от дубликатов
    
    Наследует базовый класс DataProcessing и реализует:
    - удаление дублирующихся строк
    - сброс индексов после удаления
    """
    
    @decorator
    def run(self) -> pd.DataFrame:
        """
        Выполняет очистку данных
        
        Возвращает:
        -----------
        pd.DataFrame
            Очищенный DataFrame без дубликатов с обновленными индексами
        """
        self.result = self.data.drop_duplicates().reset_index(drop=True)
        return self.result

    @decorator
    def info(self) -> str:
        """
        Возвращает описание операции
        
        Возвращает:
        -----------
        str
            Описание выполняемой очистки
        """
        return "Удаление дубликатов и сброс индексов"

    @decorator
    def get_answ(self) -> pd.DataFrame:
        """
        Получает результат очистки
        
        Если результат еще не вычислен, сначала выполняет очистку
        
        Возвращает:
        -----------
        pd.DataFrame
            Очищенный DataFrame
        """
        if self.result is None:
            self.run()
        return self.result
