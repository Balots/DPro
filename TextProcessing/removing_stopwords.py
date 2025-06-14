from .base import TextProcessing
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Union
from Logger import *

class RemoveStopwords(TextProcessing):
    def __init__(self, text: Union[str, List[str], pd.Series], lang: str = 'english', custom_stopwords: List[str] = None):
        super().__init__(text)
        self.lang = lang
        self.custom_stopwords = custom_stopwords or []
        self._stop_words = None
    
    @property
    def stop_words(self):
        if self._stop_words is None:
            try:
                self._stop_words = set(stopwords.words(self.lang)).union(set(self.custom_stopwords))
            except:
                logging.warning(f"Не удалось загрузить стоп-слова для языка {self.lang}")
                self._stop_words = set(self.custom_stopwords)
        return self._stop_words
    
    @decorator
    def run(self) -> Union[List[str], pd.Series]:
        if isinstance(self.original_text, str):
            tokens = word_tokenize(self.original_text.lower())
            self.processed_text = [word for word in tokens if word not in self.stop_words]
        elif isinstance(self.original_text, list):
            self.processed_text = [word for word in self.original_text if word not in self.stop_words]
        else:  # pd.Series
            self.processed_text = self.original_text.apply(
                lambda x: [word for word in (x if isinstance(x, list) else word_tokenize(x.lower())) 
                          if word not in self.stop_words]
            )
        return self.processed_text
    
    @decorator
    def info(self) -> str:
        return f"Удаление стоп-слов (язык: {self.lang})"
    
    @decorator
    def get_answ(self) -> Union[List[str], pd.Series]:
        if self.processed_text is None:
            self.run()
        return self.processed_text
