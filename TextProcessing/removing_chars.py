from .base import TextProcessing
import pandas as pd
import re
from typing import Union
from Logger import *

class RemoveHTMLTags(TextProcessing):
    def __init__(self, text: Union[str, pd.Series]):
        super().__init__(text)
    
    @decorator
    def run(self) -> Union[str, pd.Series]:
        if isinstance(self.original_text, str):
            self.processed_text = re.sub(r'<[^>]+>', '', self.original_text)
        else:
            self.processed_text = self.original_text.str.replace(r'<[^>]+>', '', regex=True)
        return self.processed_text
    
    @decorator
    def info(self) -> str:
        return "Удаление HTML-тегов из текста"
    
    @decorator
    def get_answ(self) -> Union[str, pd.Series]:
        if self.processed_text is None:
            self.run()
        return self.processed_text

class RemoveSpecialChars(TextProcessing):
    def __init__(self, text: Union[str, pd.Series], keep_punctuation: bool = True):
        super().__init__(text)
        self.keep_punctuation = keep_punctuation
    
    @decorator
    def run(self) -> Union[str, pd.Series]:
        pattern = r'[^\w\s]' if self.keep_punctuation else r'[^\w\s.,!?]'
        if isinstance(self.original_text, str):
            self.processed_text = re.sub(pattern, '', self.original_text)
        else:
            self.processed_text = self.original_text.str.replace(pattern, '', regex=True)
        return self.processed_text
    
    @decorator
    def info(self) -> str:
        return f"Удаление специальных символов{' (кроме пунктуации)' if self.keep_punctuation else ''}"
    
    @decorator
    def get_answ(self) -> Union[str, pd.Series]:
        if self.processed_text is None:
            self.run()
        return self.processed_text

class HandleNumbers(TextProcessing):
    def __init__(self, text: Union[str, pd.Series], strategy: str = 'keep'):
        """
        Параметры:
        - strategy: 'keep' (оставить), 'remove' (удалить), 'replace' (заменить на [NUM])
        """
        super().__init__(text)
        self.strategy = strategy
    
    @decorator
    def run(self) -> Union[str, pd.Series]:
        if self.strategy == 'keep':
            self.processed_text = self.original_text
        elif self.strategy == 'remove':
            if isinstance(self.original_text, str):
                self.processed_text = re.sub(r'\d+', '', self.original_text)
            else:
                self.processed_text = self.original_text.str.replace(r'\d+', '', regex=True)
        elif self.strategy == 'replace':
            if isinstance(self.original_text, str):
                self.processed_text = re.sub(r'\d+', '[NUM]', self.original_text)
            else:
                self.processed_text = self.original_text.str.replace(r'\d+', '[NUM]', regex=True)
        else:
            raise ValueError(f"Неизвестная стратегия: {self.strategy}")
        return self.processed_text
    
    @decorator
    def info(self) -> str:
        strategies = {
            'keep': "Сохранение чисел в тексте",
            'remove': "Удаление чисел из текста",
            'replace': "Замена чисел на [NUM]"
        }
        return strategies.get(self.strategy, "Обработка чисел в тексте")
    
    @decorator
    def get_answ(self) -> Union[str, pd.Series]:
        if self.processed_text is None:
            self.run()
        return self.processed_text