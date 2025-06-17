from .base import TextProcessing
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List, Union
#from Logger import *

class TokenizeText(TextProcessing):
    def __init__(self, text: Union[str, pd.Series], token_type: str = 'word', lang: str = 'english'):
        """
        Параметры:
        - token_type: 'word' (по словам), 'sentence' (по предложениям)
        - lang: язык текста
        """
        super().__init__(text)
        self.token_type = token_type
        self.lang = lang
    
    def run(self) -> Union[List[str], pd.Series]:
        if self.token_type == 'word':
            if isinstance(self.original_text, str):
                self.processed_text = word_tokenize(self.original_text, language=self.lang)
            else:
                self.processed_text = self.original_text.apply(lambda x: word_tokenize(x, language=self.lang))
        elif self.token_type == 'sentence':
            if isinstance(self.original_text, str):
                self.processed_text = sent_tokenize(self.original_text, language=self.lang)
            else:
                self.processed_text = self.original_text.apply(lambda x: sent_tokenize(x, language=self.lang))
        else:
            raise ValueError(f"Неизвестный тип токенизации: {self.token_type}")
        return self.processed_text
    

    def info(self) -> str:
        return f"Токенизация текста по {self.token_type} (язык: {self.lang})"

    def get_answ(self) -> Union[List[str], pd.Series]:
        if self.processed_text is None:
            self.run()
        return self.processed_text