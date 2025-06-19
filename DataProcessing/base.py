import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional
from flaml import AutoML
from sklearn.model_selection import train_test_split
from io.loader import DataLoader

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

    def ml_proc(self, target:str):
        x = self.data.drop(target)
        y = self.data[target]

        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task='classification', time_budget=60)
        y_pred = automl.predict(X_test)

        best_model = automl.model.estimator
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            print("Важность признаков:", feature_importances)
            # Связываем с именами признаков
            features_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            print(features_df)
        else:
            print("Модель не поддерживает feature_importances_")
        features_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        print(features_df)
        features_df.to_csv('weights.csv')
        return 0
