from Detector.textDetector import TextDetector
import sys
import nltk
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit, 
                             QComboBox, QLineEdit, QSplitter, QTableWidget, QTableWidgetItem, 
                             QStatusBar, QProgressBar, QAction, QToolBar, QGroupBox,
                             QCheckBox, QButtonGroup, QSpinBox, QDoubleSpinBox, QRadioButton)
from PyQt5.QtGui import QPalette, QColor, QTextDocument, QDoubleValidator
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Detector.Detector import Detector
from DataProcessing import CleanData, HandleMissingValues, DetectAndRemoveOutliers, NormalizeData, StandardizeData
from TextProcessing import RemoveHTMLTags, RemoveSpecialChars, HandleNumbers, TokenizeText, RemoveStopwords, NormalizeText
import matplotlib.pyplot as plt


class TextProcessor:
    def __init__(self):
        self.operations = []
        
    def add_operation(self, operation_type, **params):
        self.operations.append({
            'type': operation_type,
            'params': params
        })
    
    def execute(self, input_text):
        current_text = input_text
        for operation in self.operations:
            try:
                print(f"Applying operation: {operation['type']}")
                
                if operation['type'] == 'remove_html':
                    current_text = RemoveHTMLTags(current_text).run()
                elif operation['type'] == 'remove_special_chars':
                    current_text = RemoveSpecialChars(current_text).run()
                elif operation['type'] == 'handle_numbers':
                    current_text = HandleNumbers(current_text, 
                                              strategy=operation['params'].get('strategy', 'keep')).run()
                elif operation['type'] == 'tokenize':
                    token_type_val = 'word' if operation['params'].get('token_type', 'По словам') == 'По словам' else 'sentence'
                    current_text = TokenizeText(current_text, 
                                             token_type=token_type_val,
                                             lang=operation['params']['lang']).run()
                elif operation['type'] == 'remove_stopwords':
                    current_text = RemoveStopwords(current_text,
                                                lang=operation['params']['lang']).run()
                elif operation['type'] == 'normalize':
                    current_text = NormalizeText(current_text,
                                              method=operation['params'].get('method', 'stem'),
                                              lang=operation['params']['lang']).run()
                
                print(f"Result after {operation['type']}: {str(current_text)[:100]}...")
            except Exception as e:
                print(f"Error in {operation['type']}: {str(e)}")
        return current_text
    
class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.current_data = None
        self.current_text = None
        self.history = []
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        # Главный виджет и layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        # Панель управления с иконками
        control_panel = QHBoxLayout()
        self.btn_load = QPushButton("📊 Загрузить данные")
        self.btn_load_text = QPushButton("📝 Загрузить текст")
        self.btn_save = QPushButton("💾 Сохранить результат")
        self.btn_save.setEnabled(False)
        # Добавляем подсказки
        self.btn_load.setToolTip("Загрузить табличные данные (CSV, Excel, JSON)")
        self.btn_load_text.setToolTip("Загрузить текстовый файл (TXT, CSV, JSON)")
        self.btn_save.setToolTip("Сохранить текущие данные или текст")
        control_panel.addWidget(self.btn_load)
        control_panel.addWidget(self.btn_load_text)
        control_panel.addWidget(self.btn_save)
        
        # Табы
        self.tabs = QTabWidget()
        
        # Вкладка данных
        self.init_data_tab()
        
        # Вкладка анализа (для табличных данных)

        self.init_analysis_tab()
        
        # Вкладка очистки
        self.init_clean_tab()
        
        # Вкладка пропусков
        self.init_missing_tab()
        
        # Вкладка выбросов
        self.init_outliers_tab()

        self.init_text_processing_tab()

        
        # Вкладка масштабирования
        self.init_scaling_tab()

        # Лог операций
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        
        # Сборка главного layout
        main_layout.addLayout(control_panel)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(QLabel("Лог операций:"))
        main_layout.addWidget(self.log)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Статус бар
        self.init_status_bar()
        
        # Панель инструментов
        self.init_toolbar()

    def init_text_processing_tab(self):
        """Инициализация вкладки для обработки текста с анализом"""
        self.tab_text = QWidget()
        main_layout = QVBoxLayout(self.tab_text)
        
        # Создаем главный разделитель (вертикальный)
        main_splitter = QSplitter(Qt.Vertical)
        
        # 1. Верхняя часть - текст и настройки обработки
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Область для текста с прокруткой
        text_group = QGroupBox("Текст")
        text_layout = QVBoxLayout()
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("Введите текст или загрузите файл...")
        text_layout.addWidget(self.text_display)
        text_group.setLayout(text_layout)
        top_layout.addWidget(text_group)
        
        # Группа настроек обработки
        settings_group = QGroupBox("Настройки обработки")
        settings_layout = QHBoxLayout()
        
        # Колонка 1: Очистка
        clean_group = QGroupBox("Очистка")
        clean_layout = QVBoxLayout()
        self.cb_remove_html = QCheckBox("Удалить HTML-теги")
        self.cb_remove_special = QCheckBox("Удалить спецсимволы")
        self.cb_numbers_strategy = QComboBox()
        self.cb_numbers_strategy.addItems(["Оставить числа", "Удалить числа", "Заменить числа"])
        clean_layout.addWidget(self.cb_remove_html)
        clean_layout.addWidget(self.cb_remove_special)
        clean_layout.addWidget(QLabel("Обработка чисел:"))
        clean_layout.addWidget(self.cb_numbers_strategy)
        clean_group.setLayout(clean_layout)
        
        # Колонка 2: Токенизация
        token_group = QGroupBox("Токенизация")
        token_layout = QVBoxLayout()
        self.cb_tokenize = QComboBox()
        self.cb_tokenize.addItems(["Не токенизировать", "По словам", "По предложениям"])
        self.cb_language = QComboBox()
        self.cb_language.addItems(["english", "russian", "german", "french", "spanish"])
        
        # Добавляем выбор типа задачи
        self.cb_task_type = QComboBox()
        self.cb_task_type.addItems([
            "Автоопределение",
            "Исправление орфографии", 
            "Анализ тональности",
            "Извлечение сущностей",
            "Суммаризация",
            "Кластеризация",
            "Машинный перевод",
            "Ответы на вопросы"
        ])
        
        token_layout.addWidget(QLabel("Тип токенизации:"))
        token_layout.addWidget(self.cb_tokenize)
        token_layout.addWidget(QLabel("Язык текста:"))
        token_layout.addWidget(self.cb_language)
        token_layout.addWidget(QLabel("Тип NLP задачи:"))
        token_layout.addWidget(self.cb_task_type)
        token_group.setLayout(token_layout)
        
        # Колонка 3: Нормализация
        norm_group = QGroupBox("Нормализация")
        norm_layout = QVBoxLayout()
        self.cb_remove_stopwords = QCheckBox("Удалить стоп-слова")
        self.cb_normalization = QComboBox()
        self.cb_normalization.addItems(["Не нормализовать", "Стемминг", "Лемматизация"])
        norm_layout.addWidget(self.cb_remove_stopwords)
        norm_layout.addWidget(QLabel("Метод нормализации:"))
        norm_layout.addWidget(self.cb_normalization)
        norm_group.setLayout(norm_layout)
        
        settings_layout.addWidget(clean_group)
        settings_layout.addWidget(token_group)
        settings_layout.addWidget(norm_group)
        settings_group.setLayout(settings_layout)
        top_layout.addWidget(settings_group)
        
        # Кнопки обработки
        button_layout = QHBoxLayout()
        self.btn_process_text = QPushButton("Обработать текст")
        self.btn_analyze_text = QPushButton("Анализировать текст")
        button_layout.addWidget(self.btn_process_text)
        button_layout.addWidget(self.btn_analyze_text)
        top_layout.addLayout(button_layout)
        
        # 2. Нижняя часть - результаты анализа
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        
        analysis_group = QGroupBox("Результаты анализа")
        analysis_layout = QVBoxLayout()
        self.text_analysis_display = QTextEdit()
        self.text_analysis_display.setReadOnly(True)
        self.text_analysis_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        analysis_layout.addWidget(self.text_analysis_display)
        analysis_group.setLayout(analysis_layout)
        bottom_layout.addWidget(analysis_group)
        
        # Добавляем виджеты в разделитель
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setSizes([500, 300])  # Начальные размеры областей
        
        # Добавляем разделитель в главный layout
        main_layout.addWidget(main_splitter)
        
        # Устанавливаем растягивание
        main_layout.setStretchFactor(main_splitter, 1)
        
        self.tabs.addTab(self.tab_text, "📝 Текст")

    def init_data_tab(self):
        self.tab_data = QWidget()
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)

        
        # Улучшаем таблицу
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                alternate-background-color: #f8f8f8;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #c0c0c0;
            }
        """)
        
        data_layout = QVBoxLayout(self.tab_data)
        data_layout.addWidget(self.data_table)
        self.tab_data.setLayout(data_layout)
        self.tabs.addTab(self.tab_data, "📊 Данные")

    def init_analysis_tab(self):
        self.tab_analyze = QWidget()
        layout = QVBoxLayout(self.tab_analyze)
        # Кнопки анализа
        btn_panel = QHBoxLayout()
        self.btn_analyze = QPushButton("🔍 Автоанализ")
        self.btn_plot_dist = QPushButton("📊 Распределение")

        self.btn_analyze.setToolTip("Выполнить комплексный анализ данных")
        self.btn_plot_dist.setToolTip("Построить распределение для числовых столбцов")

        btn_panel.addWidget(self.btn_analyze)
        btn_panel.addWidget(self.btn_plot_dist)
        
        # Графическая область
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.canvas.setMinimumHeight(300)
        
        # Отчёт
        self.analysis_report = QTextEdit()
        self.analysis_report.setReadOnly(True)

        self.analysis_report.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        layout.addLayout(btn_panel)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Отчёт анализа:"))
        layout.addWidget(self.analysis_report)
        self.tab_analyze.setLayout(layout)
        self.tabs.addTab(self.tab_analyze, "📈 Анализ")
        layout.addLayout(btn_panel)
        layout.addWidget(self.canvas)
        layout.addWidget(self.analysis_report)
        
        self.tabs.addTab(self.tab_analyze, "Анализ")

    def init_clean_tab(self):
        self.tab_clean = QWidget()
        self.btn_clean = QPushButton("Удалить дубликаты")
        clean_layout = QVBoxLayout()
        clean_layout.addWidget(self.btn_clean)
        clean_layout.addStretch()
        self.tab_clean.setLayout(clean_layout)

        self.tabs.addTab(self.tab_clean, "🧹Очистка")


    def init_missing_tab(self):
        self.tab_missing = QWidget()
        
        self.cb_num_strategy = QComboBox()
        self.cb_num_strategy.addItems(["mean", "median", "constant"])
        
        self.cb_cat_strategy = QComboBox()
        self.cb_cat_strategy.addItems(["mode", "constant"])
        
        self.le_fill_value = QLineEdit("NULL")
        self.le_fill_value.setPlaceholderText("Значение для 'constant'")
        
        self.btn_process_missing = QPushButton("Обработать пропуски")
        
        missing_layout = QVBoxLayout()
        missing_layout.addWidget(QLabel("Стратегия для чисел:"))
        missing_layout.addWidget(self.cb_num_strategy)
        missing_layout.addWidget(QLabel("Стратегия для категорий:"))
        missing_layout.addWidget(self.cb_cat_strategy)
        missing_layout.addWidget(QLabel("Кастомное значение:"))
        missing_layout.addWidget(self.le_fill_value)
        missing_layout.addWidget(self.btn_process_missing)
        missing_layout.addStretch()
        
        self.tab_missing.setLayout(missing_layout)

        self.tabs.addTab(self.tab_missing, "❓Пропуски")


    def init_outliers_tab(self):
        self.tab_outliers = QWidget()
        
        self.le_outlier_cols = QLineEdit()
        self.le_outlier_cols.setPlaceholderText("Укажите столбцы через запятую")
        
        self.cb_outlier_method = QComboBox()
        self.cb_outlier_method.addItems(["IQR", "Hampel", "Percentile", "Skewness", "Kurtosis"])
        
        self.btn_remove_outliers = QPushButton("Удалить выбросы")
        
        outliers_layout = QVBoxLayout()
        outliers_layout.addWidget(QLabel("Столбцы для обработки:"))
        outliers_layout.addWidget(self.le_outlier_cols)
        outliers_layout.addWidget(QLabel("Метод обнаружения:"))
        outliers_layout.addWidget(self.cb_outlier_method)
        outliers_layout.addWidget(self.btn_remove_outliers)
        outliers_layout.addStretch()
        
        self.tab_outliers.setLayout(outliers_layout)

        self.tabs.addTab(self.tab_outliers, "📈Выбросы")

    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

    def init_toolbar(self):
        toolbar = self.addToolBar("Инструменты")
        
        # Действия
        export_action = QAction("Экспорт PDF", self)
        export_action.triggered.connect(self.export_report)
        
        undo_action = QAction("Отменить", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        
        toolbar.addAction(export_action)
        toolbar.addAction(undo_action)

    def setup_connections(self):
        self.btn_load.clicked.connect(self.load_data)
        self.btn_save.clicked.connect(self.save_data)

        self.btn_load_text.clicked.connect(self.load_text_file)

        self.btn_clean.clicked.connect(self.clean_data)
        self.btn_process_missing.clicked.connect(self.process_missing)
        self.btn_remove_outliers.clicked.connect(self.remove_outliers)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_plot_dist.clicked.connect(self.plot_distribution)

        self.btn_process_text.clicked.connect(self.process_text)
        self.btn_analyze_text.clicked.connect(self.analyze_text_data)


    def auto_detect_language(self):
        """Автоматическое определение языка текста"""
        if not self.current_text:
            return
            
        try:
            # Используем TextDetector для определения языка
            detector = TextDetector()
            detected_lang = detector._detect_language(self.current_text)
            
            # Устанавливаем соответствующий язык в выпадающем списке
            lang_mapping = {
                'english': 'english',
                'russian': 'russian',
                'german': 'german',
                'french': 'french',
                'spanish': 'spanish'
            }
            
            # Если определенный язык есть в нашем списке, выбираем его
            if detected_lang in lang_mapping.values():
                index = self.cb_language.findText(detected_lang)
                if index >= 0:
                    self.cb_language.setCurrentIndex(index)
                    self.log_message(f"Автоматически определен язык: {detected_lang}")
            else:
                self.log_message(f"Определен язык: {detected_lang}, но он не поддерживается")
        except Exception as e:
            self.log_message(f"Ошибка определения языка: {str(e)}")

    def load_data(self):
        """Сохранение данных в различных форматах"""
        if self.current_data is not None or self.current_text is not None:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, 
                "Сохранить результат", 
                "", 
                "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return
                
            try:
                # Добавляем расширение, если его нет
                ext_map = {
                    'Text Files (*.txt)': '.txt',
                    'CSV Files (*.csv)': '.csv',
                    'Excel Files (*.xlsx)': '.xlsx',
                    'JSON Files (*.json)': '.json'
                }
                
                if selected_filter in ext_map and not file_path.endswith(ext_map[selected_filter]):
                    file_path += ext_map[selected_filter]
                
                # Сохранение табличных данных
                if self.current_data is not None and file_path.endswith(('.csv', '.xlsx', '.json')):
                    if file_path.endswith('.csv'):
                        self.current_data.to_csv(file_path, index=False, encoding='utf-8')
                    elif file_path.endswith('.xlsx'):
                        self.current_data.to_excel(file_path, index=False)
                    elif file_path.endswith('.json'):
                        self.current_data.to_json(file_path, orient='records', indent=2, force_ascii=False)
                    self.log_message(f"Данные сохранены в {file_path}")
                
                # Сохранение текстовых данных
                elif self.current_text is not None:
                    if file_path.endswith('.txt'):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(self.current_text)
                    elif file_path.endswith('.csv'):
                        # Сохраняем текст как CSV с одной колонкой
                        lines = self.current_text.split('\n')
                        pd.DataFrame({'text': lines}).to_csv(file_path, index=False, encoding='utf-8')
                    elif file_path.endswith('.json'):
                        # Пытаемся сохранить как JSON, если текст в JSON-формате
                        try:
                            data = json.loads(self.current_text)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                        except:
                            # Если не JSON, сохраняем как текст в JSON-формате
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump({'content': self.current_text}, f, indent=2, ensure_ascii=False)
                    else:
                        # Для других форматов просто сохраняем как текст
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(self.current_text)
                    
                    self.log_message(f"Текст сохранен в {file_path}")
                    
            except Exception as e:
                self.log_message(f"Ошибка сохранения: {str(e)}", error=True)

    def process_text(self):
        """Обработка текста с учетом типа задачи"""
        if not self.current_text:
            self.log_message("Нет текста для обработки", error=True)
            return
            
        try:
            self.show_progress(True)
            
            # Сохраняем исходный текст для истории
            original_text = self.current_text
            
            # Получаем выбранные настройки из интерфейса
            selected_lang = self.cb_language.currentText()
            
            # Маппинг типа задачи
            task_type_mapping = {
                "Автоопределение": "unknown",
                "Исправление орфографии": "spelling",
                "Анализ тональности": "sentiment",
                "Извлечение сущностей": "ner",
                "Суммаризация": "summarization",
                "Кластеризация": "clustering",
                "Машинный перевод": "translation",
                "Ответы на вопросы": "question_answering"
            }
            selected_task_type = task_type_mapping.get(self.cb_task_type.currentText(), "unknown")
            
            # Создаем процессор и добавляем выбранные операции
            processor = TextProcessor()
            
            # Добавляем операции
            if self.cb_remove_html.isChecked():
                processor.add_operation('remove_html')
                
            if self.cb_remove_special.isChecked():
                processor.add_operation('remove_special_chars')
                
            if self.cb_numbers_strategy.currentText() != "Оставить числа":
                strategy = 'remove' if self.cb_numbers_strategy.currentText() == "Удалить числа" else 'replace'
                processor.add_operation('handle_numbers', strategy=strategy)
                
            if self.cb_tokenize.currentText() != "Не токенизировать":
                token_type = self.cb_tokenize.currentText()  # "По словам" или "По предложениям"
                processor.add_operation('tokenize', 
                                    token_type=token_type,
                                    lang=selected_lang)
                
            if self.cb_remove_stopwords.isChecked():
                processor.add_operation('remove_stopwords',
                                    lang=selected_lang)
                
            if self.cb_normalization.currentText() != "Не нормализовать":
                method = 'stem' if self.cb_normalization.currentText() == "Стемминг" else 'lemmatize'
                processor.add_operation('normalize',
                                    method=method,
                                    lang=selected_lang)
            
            # Выполняем все операции
            processed_result = processor.execute(str(self.current_text))
            
            # Обновляем текущий текст
            self.current_text = processed_result
            
            # Отображаем результат
            if isinstance(processed_result, (list, pd.Series)):
                # Для токенизированного текста делаем красивое отображение
                display_text = "\n".join([f"- {token}" for token in processed_result]) if isinstance(processed_result, list) else processed_result.to_string()
                self.text_display.setPlainText(display_text)
            else:
                self.text_display.setPlainText(str(self.current_text))
                
            self.log_message("Текст успешно обработан")
            
            # Сохраняем в историю
            self.history.append({
                'original': original_text,
                'processed': self.current_text
            })
            
        except Exception as e:
            self.log_message(f"Ошибка обработки: {str(e)}", error=True)
        finally:
            self.show_progress(False)

    # Новый метод для анализа текста
    def analyze_text_data(self):
        """Анализ текста и вывод рекомендаций с учетом типа задачи"""
        if not self.current_text:
            self.log_message("Нет текста для анализа", error=True)
            return
        
        try:
            self.show_progress(True)
            
            # Получаем выбранные настройки из интерфейса
            selected_lang = self.cb_language.currentText()
            
            # Маппинг выбранного типа задачи на значения TextDetector
            task_type_mapping = {
                "Автоопределение": "unknown",
                "Исправление орфографии": "spelling",
                "Анализ тональности": "sentiment",
                "Извлечение сущностей": "ner",
                "Суммаризация": "summarization",
                "Кластеризация": "clustering",
                "Машинный перевод": "translation",
                "Ответы на вопросы": "question_answering"
            }
            selected_task_type = task_type_mapping.get(self.cb_task_type.currentText(), "unknown")
            
            # Создаем анализатор текста с учетом типа задачи
            analyzer = TextDetector(lang=selected_lang, task_type=selected_task_type)
            
            # Анализируем текст
            analysis_result = analyzer.analyze_text(str(self.current_text))
            
            # Формируем отчет
            report = "=== Анализ текста ===\n\n"
            report += f"📏 Длина текста: {analysis_result['length']} символов\n"
            report += f"🌍 Определенный язык: {analysis_result['detected_lang']}\n"
            report += f"⚙️ Выбранный язык: {selected_lang}\n"
            report += f"🎯 Тип задачи: {self.cb_task_type.currentText()}\n"
            
            if 'stopword_ratio' in analysis_result and analysis_result['stopword_ratio'] is not None:
                ratio = analysis_result['stopword_ratio']
                report += f"🛑 Доля стоп-слов: {ratio:.1%} ({ratio:.4f})\n"
            
            if 'stem_vs_lemma' in analysis_result and analysis_result['stem_vs_lemma'] is not None:
                diff = analysis_result['stem_vs_lemma']['stem_diff']
                total = analysis_result['stem_vs_lemma']['total_words']
                report += f"🔤 Различия стемминга/лемматизации: {diff} из {total} слов ({diff/total:.1%})\n"
            
            report += "\n💡 Рекомендации:\n"
            for i, rec in enumerate(analysis_result.get('recommendations', []), 1):
                # Используем action_ru вместо action
                report += f"{i}. {rec['description']}\n   → Действие: {rec.get('action_ru', rec['action'])}\n"
            # Добавляем статистику
            if 'word_stats' in analysis_result:
                stats = analysis_result['word_stats']
                report += "\n📊 Статистика:\n"
                report += f" - Уникальных слов: {stats['unique_words']}\n"
                report += f" - Средняя длина слова: {stats['avg_word_length']:.2f} символов\n"
                report += f" - Частые слова: {', '.join(stats['frequent_words'][:5])}\n"
            
            # Выводим отчет в новое окно анализа
            self.text_analysis_display.setPlainText(report)
            self.log_message("Анализ текста завершен")
            
        except Exception as e:
            self.log_message(f"Ошибка анализа текста: {str(e)}", error=True)
        finally:
            self.show_progress(False)

    def load_text_file(self):
        """Загрузка текстового файла с автоматическим определением языка"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите текстовый файл", 
            "", 
            "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Очищаем текущий текст
                self.current_text = None
                
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.current_text = f.read()
                
                elif file_path.endswith('.csv'):
                    # Читаем CSV, преобразуем в удобный текстовый формат
                    df = pd.read_csv(file_path)
                    # Формируем текст из названий столбцов и данных
                    columns = " | ".join(df.columns)
                    data_rows = []
                    for _, row in df.iterrows():
                        row_str = " | ".join(str(value) for value in row.values)
                        data_rows.append(row_str)
                    
                    self.current_text = f"Columns: {columns}\n\n" + "\n".join(data_rows)
                    
                elif file_path.endswith('.xlsx'):
                    # Читаем Excel, преобразуем в удобный текстовый формат
                    df = pd.read_excel(file_path)
                    columns = " | ".join(df.columns)
                    data_rows = []
                    for _, row in df.iterrows():
                        row_str = " | ".join(str(value) for value in row.values)
                        data_rows.append(row_str)
                    
                    self.current_text = f"Columns: {columns}\n\n" + "\n".join(data_rows)
                    
                elif file_path.endswith('.json'):
                    # Читаем JSON, преобразуем в текст
                    import json  # Добавляем импорт json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self.current_text = "\n".join(f"{k}: {v}" for k, v in data.items())
                        elif isinstance(data, list):
                            self.current_text = "\n".join(str(item) for item in data)
                        else:
                            self.current_text = str(data)
                
                if self.current_text is not None:
                    self.text_display.setPlainText(self.current_text)
                    self.btn_save.setEnabled(True)
                    self.log_message(f"Текст загружен из {file_path}")
                    
                    # Автоматическое определение языка
                    self.auto_detect_language()
                    
            except Exception as e:
                self.log_message(f"Ошибка загрузки файла: {str(e)}", error=True)

    def save_data(self):
        """Сохранение данных в различных форматах"""
        if self.current_data is not None or self.current_text is not None:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, 
                "Сохранить результат", 
                "", 
                "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return
                
            try:
                # Добавляем импорт json в начало метода
                import json
                
                # Добавляем расширение, если его нет
                ext_map = {
                    'Text Files (*.txt)': '.txt',
                    'CSV Files (*.csv)': '.csv',
                    'Excel Files (*.xlsx)': '.xlsx',
                    'JSON Files (*.json)': '.json'
                }
                
                if selected_filter in ext_map and not file_path.endswith(ext_map[selected_filter]):
                    file_path += ext_map[selected_filter]
                
                # Сохранение табличных данных
                if self.current_data is not None and file_path.endswith(('.csv', '.xlsx', '.json')):
                    if file_path.endswith('.csv'):
                        self.current_data.to_csv(file_path, index=False, encoding='utf-8')
                    elif file_path.endswith('.xlsx'):
                        self.current_data.to_excel(file_path, index=False)
                    elif file_path.endswith('.json'):
                        self.current_data.to_json(file_path, orient='records', indent=2, force_ascii=False)
                    self.log_message(f"Данные сохранены в {file_path}")
                
                # Сохранение текстовых данных
                elif self.current_text is not None:
                    if file_path.endswith('.txt'):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(self.current_text)
                    elif file_path.endswith('.csv'):
                        # Разбиваем текст на строки и сохраняем как CSV
                        lines = [line.strip() for line in self.current_text.split('\n') if line.strip()]
                        pd.DataFrame({'text': lines}).to_csv(file_path, index=False, encoding='utf-8')
                    elif file_path.endswith('.json'):
                        # Пытаемся интерпретировать текст как JSON
                        try:
                            data = json.loads(self.current_text)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                        except json.JSONDecodeError:
                            # Если не JSON, сохраняем как текст в JSON-формате
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump({'content': self.current_text}, f, indent=2, ensure_ascii=False)
                    elif file_path.endswith('.xlsx'):
                        # Сохраняем текст в Excel (каждую строку в отдельную строку таблицы)
                        lines = [line.strip() for line in self.current_text.split('\n') if line.strip()]
                        pd.DataFrame({'text': lines}).to_excel(file_path, index=False)
                    else:
                        # Для других форматов просто сохраняем как текст
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(self.current_text)
                    
                    self.log_message(f"Текст сохранен в {file_path}")
                    
            except Exception as e:
                self.log_message(f"Ошибка сохранения: {str(e)}", error=True)


    def display_data(self):
        if self.current_data is not None:
            self.data_table.setRowCount(self.current_data.shape[0])
            self.data_table.setColumnCount(self.current_data.shape[1])
            self.data_table.setHorizontalHeaderLabels(self.current_data.columns)
            
            for row in range(self.current_data.shape[0]):
                for col in range(self.current_data.shape[1]):
                    item = QTableWidgetItem(str(self.current_data.iloc[row, col]))
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    self.data_table.setItem(row, col, item)

    def clean_data(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                processor = CleanData(self.current_data)
                self.current_data = processor.run()
                self.display_data()
                self.save_state()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка очистки: {str(e)}", error=True)
            finally:
                self.show_progress(False)

    def process_missing(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                fill_value = {col: self.le_fill_value.text() 
                             for col in self.current_data.columns}
                
                processor = HandleMissingValues(
                    self.current_data,
                    numeric_strategy=self.cb_num_strategy.currentText(),
                    categorical_strategy=self.cb_cat_strategy.currentText(),
                    fill_value=fill_value
                )
                self.current_data = processor.run()
                self.display_data()
                self.save_state()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка обработки пропусков: {str(e)}", error=True)
            finally:
                self.show_progress(False)

    def remove_outliers(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                columns = [col.strip() for col in self.le_outlier_cols.text().split(",")] if self.le_outlier_cols.text() else None
                
                processor = DetectAndRemoveOutliers(
                    self.current_data, 
                    columns=columns,
                    method=self.cb_outlier_method.currentText().lower()
                )
                self.current_data = processor.run()
                self.display_data()
                self.save_state()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка удаления выбросов: {str(e)}", error=True)
            finally:
                self.show_progress(False)

    def init_scaling_tab(self):
        """Вкладка для масштабирования данных"""
        self.tab_scaling = QWidget()
        layout = QVBoxLayout()
        
        # Выбор столбцов
        self.scaling_columns = QLineEdit()
        self.scaling_columns.setPlaceholderText("Укажите столбцы через запятую (оставьте пустым для всех числовых)")
        
        # Группа методов
        self.scaling_method = QButtonGroup()
        self.rb_normalize = QRadioButton("Нормализация (MinMax)")
        self.rb_standardize = QRadioButton("Стандартизация (Z-score)")
        self.rb_normalize.setChecked(True)
        self.scaling_method.addButton(self.rb_normalize)
        self.scaling_method.addButton(self.rb_standardize)
        
        # Параметры нормализации
        self.norm_range_layout = QHBoxLayout()
        self.norm_range_layout.addWidget(QLabel("Диапазон:"))
        self.norm_min = QLineEdit("0")
        self.norm_max = QLineEdit("1")
        self.norm_min.setValidator(QDoubleValidator())
        self.norm_max.setValidator(QDoubleValidator())
        self.norm_range_layout.addWidget(self.norm_min)
        self.norm_range_layout.addWidget(QLabel("до"))
        self.norm_range_layout.addWidget(self.norm_max)
        
        # Контейнер для параметров нормализации
        self.norm_params_container = QWidget()
        self.norm_params_container.setLayout(self.norm_range_layout)
        
        # Кнопка выполнения
        self.btn_apply_scaling = QPushButton("Применить масштабирование")
        
        # Сборка layout
        layout.addWidget(QLabel("Столбцы для обработки:"))
        layout.addWidget(self.scaling_columns)
        layout.addWidget(QLabel("Метод:"))
        layout.addWidget(self.rb_normalize)
        layout.addWidget(self.rb_standardize)
        layout.addWidget(self.norm_params_container)
        layout.addWidget(self.btn_apply_scaling)
        layout.addStretch()
        
        # Подключение сигналов
        self.rb_normalize.toggled.connect(self.norm_params_container.setVisible)
        self.norm_params_container.setVisible(self.rb_normalize.isChecked())
        
        self.tab_scaling.setLayout(layout)
        self.tabs.addTab(self.tab_scaling, "Масштабирование")

    def apply_scaling(self):
        """Применяет выбранный метод масштабирования"""
        if self.current_data is None:
            self.log_message("Нет данных для обработки", error=True)
            return
            
        try:
            self.show_progress(True)
            columns = [c.strip() for c in self.scaling_columns.text().split(",")] if self.scaling_columns.text() else None
            
            if self.rb_normalize.isChecked():
                processor = NormalizeData(
                    self.current_data,
                    columns=columns,
                    feature_range=(
                        float(self.norm_min.text()),
                        float(self.norm_max.text())
                    ))
            else:
                processor = StandardizeData(
                    self.current_data,
                    columns=columns
                )
                
            self.current_data = processor.run()
            self.display_data()
            self.save_state()
            self.log_message(processor.info())
            
        except ValueError as e:
            self.log_message(f"Ошибка ввода параметров: {str(e)}", error=True)
        except Exception as e:
            self.log_message(f"Ошибка масштабирования: {str(e)}", error=True)
        finally:
            self.show_progress(False)

    # Новые методы анализа

    def run_analysis(self):
        if self.current_data is not None:
            try:
                self.show_progress(True)
                temp_file = "temp_analysis.csv"
                self.current_data.to_csv(temp_file, index=False)
                
                detector = Detector(
                    check_abnormal=True,
                    check_missing=True,
                    check_duplicates=True,
                    check_scaling=True
                )
                outcome, abnormal, scaling = detector.check_dataframe(temp_file)
                
                report = "=== Анализ данных ===\n"
                report += f"Пропуски: {outcome['Missing values/Пропущенные значения']}\n"
                report += f"Дубликаты: {outcome['Duplicate values/Дубликаты значений ']}\n\n"
                report += "Рекомендации по масштабированию:\n"
                for col, rec in scaling.items():
                    report += f"- {col}: {rec['Рекомендация']} ({', '.join(rec['причина'])})\n"
                
                self.analysis_report.setPlainText(report)
                self.plot_distribution()
                self.log_message("Автоанализ завершен")
                
            except Exception as e:
                self.log_message(f"Ошибка анализа: {str(e)}", error=True)
            finally:
                self.show_progress(False)

    def plot_distribution(self):
        if self.current_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            numeric_cols = self.current_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                self.current_data[col].plot(kind='hist', ax=ax)
                ax.set_title(f"Распределение {col}")
                self.canvas.draw()

    # Система истории и другие утилиты

    def save_state(self):
        if self.current_data is not None:
            self.history.append(self.current_data.copy())
            if len(self.history) > 10:
                self.history.pop(0)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.current_data = self.history[-1].copy()
            self.display_data()
            self.log_message("Отмена последнего действия")

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт отчёта", "", "PDF Files (*.pdf)")
        if path:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            
            doc = QTextDocument()
            doc.setPlainText(self.analysis_report.toPlainText())
            doc.print_(printer)
            self.log_message(f"Отчёт экспортирован в {path}")

    def show_progress(self, visible):
        self.progress_bar.setVisible(visible)
        self.progress_bar.setRange(0, 0 if visible else 1)  # Неопределённый прогресс
        QApplication.processEvents()

    def log_message(self, message, error=False):
        if error:
            self.log.append(f"<font color='red'>{message}</font>")
        else:
            self.log.append(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Настройка стиля
    app.setStyle('Fusion')
    
    # Улучшенная цветовая схема
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(70, 130, 180))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    # Настройка шрифтов
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())

