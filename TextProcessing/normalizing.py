from .base import TextProcessing
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Union
from Logger import *

class NormalizeText(TextProcessing):
    def __init__(self, text: Union[str, List[str], pd.Series], method: str = 'lemmatize', lang: str = 'english'):
        """
        Параметры:
        - method: 'stem' (стемминг), 'lemmatize' (лемматизация)
        - lang: язык текста
        """
        super().__init__(text)
        self.method = method
        self.lang = lang
        self._stemmer = None
        self._lemmatizer = None
    
    @property
    def stemmer(self):
        if self._stemmer is None:
            self._stemmer = PorterStemmer()
        return self._stemmer
    
    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer
    
    @decorator
    def run(self) -> Union[List[str], pd.Series]:
        if self.method == 'stem':
            if isinstance(self.original_text, str):
                tokens = word_tokenize(self.original_text.lower())
                self.processed_text = [self.stemmer.stem(word) for word in tokens]
            elif isinstance(self.original_text, list):
                self.processed_text = [self.stemmer.stem(word) for word in self.original_text]
            else:  # pd.Series
                self.processed_text = self.original_text.apply(
                    lambda x: [self.stemmer.stem(word) for word in (x if isinstance(x, list) else word_tokenize(x.lower()))]
                )
        elif self.method == 'lemmatize':
            if isinstance(self.original_text, str):
                tokens = word_tokenize(self.original_text.lower())
                self.processed_text = [self.lemmatizer.lemmatize(word) for word in tokens]
            elif isinstance(self.original_text, list):
                self.processed_text = [self.lemmatizer.lemmatize(word) for word in self.original_text]
            else:  # pd.Series
                self.processed_text = self.original_text.apply(
                    lambda x: [self.lemmatizer.lemmatize(word) for word in (x if isinstance(x, list) else word_tokenize(x.lower()))]
                )
        else:
            raise ValueError(f"Неизвестный метод нормализации: {self.method}")
        return self.processed_text
    
    @decorator
    def info(self) -> str:
        methods = {
            'stem': "Стемминг текста",
            'lemmatize': "Лемматизация текста"
        }
        return f"{methods.get(self.method, 'Нормализация текста')} (язык: {self.lang})"
    
    @decorator
    def get_answ(self) -> Union[List[str], pd.Series]:
        if self.processed_text is None:
            self.run()
        return self.processed_text