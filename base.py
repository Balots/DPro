import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional
from .io.loader import DataLoader

class DataProcessing(ABC):
    def __init__(self, data: Union[pd.DataFrame, str], file_type: Optional[str] = None):
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            self.data = DataLoader.load_data(data, file_type)
        else:
            raise ValueError("Data must be either a DataFrame or a file path")
            
        self.result = None

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute data processing"""
        pass

    @abstractmethod
    def info(self) -> str:
        """Description of the processing"""
        pass

    @abstractmethod
    def get_answ(self) -> pd.DataFrame:
        """Get processing result"""
        pass

    def _select_numeric_columns(self) -> list:
        """Helper method to select numeric columns"""
        return self.data.select_dtypes(include='number').columns.tolist()

    def save_result(self, file_path: str, file_type: Optional[str] = None, **kwargs) -> None:
        """Save processing result to file"""
        if self.result is None:
            self.run()
        DataLoader.save_data(self.result, file_path, file_type, **kwargs)
