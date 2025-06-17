from .base import TextProcessing
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import download
from typing import Union
import pandas as pd
import logging
from natasha import MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger


class NormalizeText(TextProcessing):
    def __init__(self, text: Union[str, pd.Series], method: str = 'stem', lang: str = 'russian'):
        super().__init__(text)
        self.method = method
        self.lang = lang
        
        # Инициализация процессоров
        if lang == 'russian':
            if method == 'lemmatize':
                self.segmenter = Segmenter()
                self.emb = NewsEmbedding()
                self.morph_tagger = NewsMorphTagger(self.emb)
                self.morph_vocab = MorphVocab()
            else:  # stem
                self.stemmer = SnowballStemmer('russian')
        else:  # english
            if method == 'lemmatize':
                self.lemmatizer = WordNetLemmatizer()
            else:  # stem
                self.stemmer = SnowballStemmer('english')

    def _process_russian(self, text: str) -> str:
        """Обработка русского текста"""
        try:
            if self.method == 'stem':
                tokens = word_tokenize(text.lower(), language='russian')
                return ' '.join(self.stemmer.stem(word) for word in tokens)
            else:  # lemmatize
                from natasha import Doc
                doc = Doc(text)
                doc.segment(self.segmenter)
                doc.tag_morph(self.morph_tagger)
                
                for token in doc.tokens:
                    token.lemmatize(self.morph_vocab)
                
                tokens = [token.lemma for token in doc.tokens if token.lemma is not None]
                return ' '.join(tokens)
        except Exception as e:
            logging.error(f"Russian processing error: {str(e)}", exc_info=True)
            return text

    def _process_english(self, text: str) -> str:
        """Обработка английского текста"""
        try:
            tokens = word_tokenize(text.lower(), language='english')
            if self.method == 'stem':
                return ' '.join(self.stemmer.stem(word) for word in tokens)
            else:  # lemmatize
                return ' '.join(self.lemmatizer.lemmatize(word) for word in tokens)
        except Exception as e:
            logging.error(f"English processing error: {str(e)}")
            return text

    def run(self) -> Union[str, pd.Series]:
        try:
            if isinstance(self.original_text, str):
                if self.lang == 'russian':
                    self.processed_text = self._process_russian(self.original_text)
                else:
                    self.processed_text = self._process_english(self.original_text)
            elif isinstance(self.original_text, pd.Series):
                if self.lang == 'russian':
                    self.processed_text = self.original_text.apply(self._process_russian)
                else:
                    self.processed_text = self.original_text.apply(self._process_english)
            else:
                raise ValueError("Unsupported input type")
            return self.processed_text
        except Exception as e:
            logging.error(f"Normalization error: {str(e)}")
            return self.original_text

    def info(self) -> str:
        methods = {
            'stem': 'стемминг',
            'lemmatize': 'лемматизация'
        }
        return f"Нормализация ({methods[self.method]}, {self.lang})"

    def get_answ(self) -> Union[str, pd.Series]:
        return self.processed_text if self.processed_text is not None else self.original_text
