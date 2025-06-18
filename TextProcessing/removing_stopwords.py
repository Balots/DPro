from .base import TextProcessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import pandas as pd
from typing import Union
import logging

# Скачиваем необходимые ресурсы NLTK
download('stopwords', quiet=True)
download('punkt', quiet=True)

class RemoveStopwords(TextProcessing):
    def __init__(self, text: Union[str, pd.Series], lang: str = 'russian'):
        """
        Инициализация обработчика стоп-слов
        :param text: Входной текст (строка или pandas.Series)
        :param lang: Язык текста ('russian' или 'english')
        """
        super().__init__(text)
        self.lang = lang
        self.stop_words = self._load_stopwords()
        
    def _load_stopwords(self) -> set:
        """Загрузка стоп-слов для указанного языка"""
        try:
            return set(stopwords.words(self.lang))
        except Exception as e:
            logging.error(f"Error loading stopwords for {self.lang}: {str(e)}")
            return set()

    def _process_text(self, text: str) -> str:
        """Обработка одного текстового фрагмента (без стемминга!)"""
        tokens = word_tokenize(text.lower())
        # Только фильтрация стоп-слов, без изменения самих слов
        filtered_words = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_words)

    def run(self) -> Union[str, pd.Series]:
        """Основной метод обработки текста"""
        try:
            if isinstance(self.original_text, str):
                self.processed_text = self._process_text(self.original_text)
            elif isinstance(self.original_text, pd.Series):
                self.processed_text = self.original_text.apply(self._process_text)
            else:
                raise ValueError("Unsupported input type")
            return self.processed_text
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return self.original_text

    def info(self) -> str:
        """Информация о выполненной обработке"""
        return f"Removed {self.lang} stopwords (without stemming)"

    def get_answ(self) -> Union[str, pd.Series]:
        """Получение результата обработки"""
        return self.processed_text if self.processed_text is not None else self.original_text