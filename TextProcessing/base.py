import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
#from Logger import *
    
class TextProcessing(ABC):
    def __init__(self, text: Union[str, pd.Series]):
        """
        Базовый класс для обработки текста
        """
        self.original_text = text
        self.processed_text = None
    
    @abstractmethod
    def run(self) -> Union[str, pd.Series]:
        pass
    
    @abstractmethod
    def info(self) -> str:
        pass
    
    @abstractmethod
    def get_answ(self) -> Union[str, pd.Series]:
        pass