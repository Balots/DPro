import pandas as pd
import re
from langdetect import detect, DetectorFactory
import json
import sys
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import Dict, Union, List
import numpy as np
from pathlib import Path
#from Logger import *

class TextDetector:
    def __init__(self, 
                 task_type: str = 'unknown',
                 check_clean: bool = True,
                 check_tokenize: bool = True,
                 check_stopwords: bool = True,
                 check_stem_lemm: bool = True,
                 lang: str = None): 
        """
        Параметры:
        - lang: язык текста ('english', 'russian' и т.д.)
        - task_type: тип NLP-задачи, для которой будет использоваться токенизация.
            Влияет на рекомендации. 
            Допустимые значения:
            - 'spelling'       - Исправление орфографии.
            - 'sentiment'      - Анализ тональности.
            - 'ner'            - Извлечение именованных сущностей.
            - 'summarization'  - Суммаризация текста.
            - 'clustering'     - Кластеризация документов по темам.
            - 'translation'    - Машинный перевод.
            - 'question_answering' - Ответы на вопросы по тексту.
            - 'unknown'        - Автоматический подбор (по умолчанию).
        """
        self.task_type = task_type
        self.check_clean = check_clean
        self.check_tokenize = check_tokenize
        self.check_stopwords = check_stopwords
        self.check_stem_lemm = check_stem_lemm
        self.lang = lang
        # Ленивая инициализация
        self._stop_words = None
        self._ps = None
        self._lemmatizer = None
        self._token_cache = {}
        
    @property
    def stop_words(self):
        if self._stop_words is None:
            try:
                self._stop_words = set(stopwords.words(self.lang))
            except:
                print(f"[WARNING] Не удалось загрузить стоп-слова для языка {self.lang}", file=sys.stderr)
                self._stop_words = set()
        return self._stop_words
    
    @property
    def ps(self):
        if self._ps is None:
            self._ps = PorterStemmer()
        return self._ps
    
    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer
    
    def load_text_from_file(self, file_path: Union[str, Path]) -> str:
        """Оптимизированная загрузка файлов с кэшированием"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл {file_path} не найден")
            
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_path.suffix == '.csv':
            # Используем только нужные колонки
            df = pd.read_csv(file_path, usecols=[0])
            return ' '.join(df.iloc[:, 0].astype(str).tolist())
            
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return ' '.join(str(v) for v in data.values())
                return ' '.join(str(item) for item in data)
                
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")
    
    def analyze_text(self, text: Union[str, pd.Series]) -> Dict:
        """Оптимизированный анализ с кэшированием промежуточных результатов"""
        if isinstance(text, pd.Series):
            return self._analyze_series(text)
        
        # Кэшируем токенизацию
        words = self._get_cached_tokenization(text, 'words')
        sentences = self._get_cached_tokenization(text, 'sentences')
        
        metrics = {
            'length': len(text),
            'detected_lang': self._detect_language(text),
            'stopword_ratio': self._stopword_ratio(words, self._detect_language(text)) if self.check_stopwords else None,
            'stem_vs_lemma': self._compare_stem_lemm(words) if self.check_stem_lemm else None,
            'words': words,
            'sentences': sentences
        }
        
        metrics['recommendations'] = self._generate_recommendations(text, metrics)
        return metrics

    def _get_cached_tokenization(self, text: str, token_type: str) -> List[str]:
        """Кэширование результатов токенизации"""
        cache_key = (hash(text), token_type)
        if cache_key not in self._token_cache:
            if token_type == 'words':
                self._token_cache[cache_key] = word_tokenize(text)
            else:
                self._token_cache[cache_key] = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return self._token_cache[cache_key]

    def _analyze_series(self, series: pd.Series) -> Dict:
        """Оптимизированный анализ серии с выборкой"""
        sample_size = min(5, len(series))
        samples = series.sample(sample_size, random_state=42)  # Фиксируем random_state для воспроизводимости
        
        analyses = []
        for text in samples:
            analysis = self.analyze_text(text)
            analyses.append(analysis)
            # Очищаем кэш после каждого анализа для экономии памяти
            self._token_cache.clear()
        
        return {
            'avg_length': np.mean([a['length'] for a in analyses]),
            'main_lang': Counter([a['detected_lang'] for a in analyses]).most_common(1)[0][0],
            'recommendations': list(set(r for a in analyses for r in a['recommendations']))
        }

    def _detect_language(self, text: str) -> str:
        """Определение языка с использованием langdetect"""
        if self.lang:  # Если язык задан явно
            return self.lang
            
        try:
            # Берем только первые 500 символов для ускорения (langdetect может ошибаться на коротких текстах)
            lang = detect(text[:500])
            return 'russian' if lang == 'ru' else 'english'  # Поддерживаем пока 2 языка
        except:
            return 'english'  # По умолчанию

    def _stopword_ratio(self, words: List[str], lang: str) -> float:
        """Точный расчет доли стоп-слов с учетом языка"""
        if not words:
            return 0.0
            
        # Обновляем стоп-слова для текущего языка
        try:
            self._stop_words = set(stopwords.words(lang))
        except:
            self._stop_words = set()
            
        stopword_count = sum(1 for word in words if word.lower() in self._stop_words)
        return stopword_count / len(words)

    def _compare_stem_lemm(self, words: List[str]) -> Dict:
        """Оптимизированное сравнение стемминга и лемматизации"""
        stemmed = [self.ps.stem(word) for word in words]
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        
        return {
            'stem_diff': sum(1 if s != l else 0 for s, l in zip(stemmed, lemmatized)),
            'total_words': len(words)
        }

    def _generate_recommendations(self, text: str, metrics: Dict) -> List[Dict]:
        """Генерация рекомендаций с русскими названиями действий"""
        recommendations = []
        if self.check_clean:
            recommendations.extend(self._get_cleaning_rec(text))
        if self.check_tokenize:
            recommendations.extend(self._get_tokenization_rec(metrics))
        if self.check_stopwords and metrics['stopword_ratio'] is not None:
            recommendations.extend(self._get_stopwords_rec(metrics['stopword_ratio']))
        if self.check_stem_lemm and metrics['stem_vs_lemma'] is not None:
            recommendations.extend(self._get_stemm_lemm_rec(metrics['stem_vs_lemma']))
        
        # Маппинг английских действий на русские
        action_translation = {
            'remove_html': 'Удалить HTML-теги',
            'remove_special_chars': 'Удалить спецсимволы',
            'handle_numbers': 'Обработать числа',
            'word_tokenize': 'Токенизировать по словам',
            'sentence_tokenize': 'Токенизировать по предложениям',
            'remove_stopwords': 'Удалить стоп-слова',
            'consider_remove_stopwords': 'Рассмотреть удаление стоп-слов',
            'lemmatize': 'Применить лемматизацию',
            'lemmatize_or_stem': 'Применить лемматизацию или стемминг',
            'stem': 'Применить стемминг'
        }
        
        # Переводим действия на русский
        for rec in recommendations:
            rec['action_ru'] = action_translation.get(rec['action'], rec['action'])
        
        return recommendations

    def _get_cleaning_rec(self, text: str) -> List[Dict]:
        """Объединенные проверки очистки текста"""
        recommendations = []
        has_html = bool(re.search(r'<[^>]+>', text))
        has_special_chars = bool(re.search(r'[^\w\s.,!?]', text))
        has_numbers = bool(re.search(r'\d', text))
        
        if has_html:
            recommendations.append({'action': 'remove_html', 'description': 'Текст содержит HTML-теги. Рекомендуется удаление'})
        if has_special_chars:
            recommendations.append({'action': 'remove_special_chars', 'description': 'Текст содержит спецсимволы. Рекомендуется удаление'})
        if has_numbers:
            recommendations.append({'action': 'handle_numbers', 'description': 'Текст содержит цифры. Рекомендуется выбрать стратегию обработки'})
        
        return recommendations

    def _get_tokenization_rec(self, metrics: Dict) -> List[Dict]:
        """Оптимизированные рекомендации по токенизации"""
        task_rules = {
            "spelling": {"method": "word_tokenize", "reason": "Точное исправление ошибок требует разбивки по словам"},
            "sentiment": {"method": "sentence_tokenize", "reason": "Контекст предложения важен для тональности"},
            "ner": {"method": "word_tokenize", "reason": "Распознавание сущностей требует анализа слов"},
            "summarization": {"method": "sentence_tokenize", "reason": "Сохранение структуры предложений"},
            "clustering": {"method": "word_tokenize", "reason": "Тематический анализ требует анализа слов"},
            "translation": {"method": "sentence_tokenize", "reason": "Контекст предложения важен для перевода"},
            "question_answering": {"method": "sentence_tokenize", "reason": "Анализ вопроса и контекста"},
            "unknown": self._get_auto_tokenization_rec(metrics)
        }
        
        rule = task_rules.get(self.task_type, task_rules["unknown"])
        
        if isinstance(rule, dict):
            return [{"action": rule["method"], "description": rule["reason"]}]
        return rule

    def _get_auto_tokenization_rec(self, metrics: Dict) -> List[Dict]:
        """Автоматические рекомендации по токенизации"""
        words = metrics.get('words', [])
        sentences = metrics.get('sentences', [])
        
        if not sentences or len(words) < 5:
            return [{"action": "word_tokenize", "description": "Короткий текст: разбивка по словам"}]
        
        avg_length = len(words) / len(sentences)
        if avg_length > 15:
            return [{"action": "sentence_tokenize", "description": "Длинные предложения (>15 слов)"}]
        elif "?" in ' '.join(sentences):
            return [{"action": "sentence_tokenize", "description": "Наличие вопросов"}]
        else:
            return [{"action": "word_tokenize", "description": "Стандартная рекомендация"}]

    def _get_stopwords_rec(self, ratio: float) -> List[Dict]:
        """Оптимизированные рекомендации по стоп-словам"""
        if ratio > 0.4:
            return [{'action': 'remove_stopwords', 'description': f'Высокий процент стоп-слов ({ratio:.0%})'}]
        elif ratio > 0.2:
            return [{'action': 'consider_remove_stopwords', 'description': f'Умеренный процент стоп-слов ({ratio:.0%})'}]
        return []

    def _get_stemm_lemm_rec(self, stem_lemm_data: Dict) -> List[Dict]:
        """Оптимизированные рекомендации по стеммингу/лемматизации"""
        diff_ratio = stem_lemm_data['stem_diff'] / stem_lemm_data['total_words'] if stem_lemm_data['total_words'] > 0 else 0
        if diff_ratio > 0.3:
            return [{'action': 'lemmatize', 'description': 'Лемматизация значительно лучше сохраняет смысл'}]
        elif diff_ratio > 0.1:
            return [{'action': 'lemmatize_or_stem', 'description': 'Лемматизация предпочтительна'}]
        return [{'action': 'stem', 'description': 'Стемминг достаточен'}]