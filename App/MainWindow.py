# MainWindow.py (дополненная версия)

import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit, 
                             QComboBox, QLineEdit, QTableWidget, QTableWidgetItem, 
                             QStatusBar, QProgressBar, QAction, QToolBar, QGroupBox,
                             QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QPalette, QColor, QTextDocument
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
        for op in self.operations:
            try:
                if op['type'] == 'remove_html':
                    current_text = RemoveHTMLTags(current_text).run()
                elif op['type'] == 'remove_special_chars':
                    current_text = RemoveSpecialChars(current_text).run()
                elif op['type'] == 'handle_numbers':
                    current_text = HandleNumbers(current_text, 
                                              strategy=op['params'].get('strategy', 'keep')).run()
                elif op['type'] == 'tokenize':
                    current_text = TokenizeText(current_text,
                                             token_type=op['params'].get('token_type', 'word'),
                                             lang=op['params'].get('lang', 'english')).run()
                elif op['type'] == 'remove_stopwords':
                    current_text = RemoveStopwords(current_text,
                                                lang=op['params'].get('lang', 'english')).run()
                elif op['type'] == 'normalize':
                    current_text = NormalizeText(current_text,
                                              method=op['params'].get('method', 'stem'),
                                              lang=op['params'].get('lang', 'english')).run()
            except Exception as e:
                print(f"Error in {op['type']}: {str(e)}")
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
        
        # Панель управления
        control_panel = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить данные")
        self.btn_load_text = QPushButton("Загрузить текст")  # Новая кнопка для текста
        self.btn_save = QPushButton("Сохранить результат")
        self.btn_save.setEnabled(False)
        control_panel.addWidget(self.btn_load)
        control_panel.addWidget(self.btn_load_text)
        control_panel.addWidget(self.btn_save)
        
        # Табы
        self.tabs = QTabWidget()
        
        # Вкладка данных
        self.init_data_tab()
        
        # Вкладка анализа
        self.init_analysis_tab()
        
        # Вкладка очистки
        self.init_clean_tab()
        
        # Вкладка пропусков
        self.init_missing_tab()
        
        # Вкладка выбросов
        self.init_outliers_tab()
        
        # Вкладка обработки текста
        self.init_text_processing_tab()
        
        # Лог операций
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        
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
        """Инициализация вкладки для обработки текста"""
        self.tab_text = QWidget()
        layout = QVBoxLayout()
        # Область для отображения текста
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        # Выбор столбца для обработки
        self.cb_text_column = QComboBox()
        self.cb_text_column.setPlaceholderText("Выберите столбец")
        
        # Группа очистки текста
        clean_group = QGroupBox("Очистка текста")
        clean_layout = QVBoxLayout()
        
        self.cb_remove_html = QCheckBox("Удалить HTML-теги")
        self.cb_remove_html.setChecked(True)
        
        self.cb_remove_special = QCheckBox("Удалить спецсимволы")
        self.cb_remove_special.setChecked(True)
        
        self.cb_numbers_strategy = QComboBox()
        self.cb_numbers_strategy.addItems(["Оставить числа", "Удалить числа"])
        
        clean_layout.addWidget(self.cb_remove_html)
        clean_layout.addWidget(self.cb_remove_special)
        clean_layout.addWidget(QLabel("Обработка чисел:"))
        clean_layout.addWidget(self.cb_numbers_strategy)
        clean_group.setLayout(clean_layout)
        
        # Группа токенизации
        token_group = QGroupBox("Токенизация")
        token_layout = QVBoxLayout()
        
        self.cb_tokenize = QComboBox()
        self.cb_tokenize.addItems(["Не токенизировать", "По словам", "По предложениям"])
        
        self.cb_language = QComboBox()
        self.cb_language.addItems(["english", "russian", "german", "french", "spanish"])
        
        token_layout.addWidget(QLabel("Тип токенизации:"))
        token_layout.addWidget(self.cb_tokenize)
        token_layout.addWidget(QLabel("Язык текста:"))
        token_layout.addWidget(self.cb_language)
        token_group.setLayout(token_layout)
        
        # Группа нормализации
        norm_group = QGroupBox("Нормализация")
        norm_layout = QVBoxLayout()
        
        self.cb_remove_stopwords = QCheckBox("Удалить стоп-слова")
        self.cb_remove_stopwords.setChecked(True)
        
        self.cb_normalization = QComboBox()
        self.cb_normalization.addItems(["Не нормализовать", "Стемминг", "Лемматизация"])
        
        norm_layout.addWidget(self.cb_remove_stopwords)
        norm_layout.addWidget(QLabel("Метод нормализации:"))
        norm_layout.addWidget(self.cb_normalization)
        norm_group.setLayout(norm_layout)
        
        # Кнопка обработки
        self.btn_process_text = QPushButton("Обработать текст")
        
        # Сборка layout
        layout.addWidget(self.text_display)
        layout.addWidget(clean_group)
        layout.addWidget(token_group)
        layout.addWidget(norm_group)
        layout.addWidget(self.btn_process_text)
        layout.addStretch()
        
        self.tab_text.setLayout(layout)
        self.tabs.addTab(self.tab_text, "Текст")

    def init_data_tab(self):
        self.tab_data = QWidget()
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        data_layout = QVBoxLayout()
        data_layout.addWidget(self.data_table)
        self.tab_data.setLayout(data_layout)
        self.tabs.addTab(self.tab_data, "Данные")

    def init_analysis_tab(self):
        self.tab_analyze = QWidget()
        layout = QVBoxLayout()
        
        # Кнопки анализа
        btn_panel = QHBoxLayout()
        self.btn_analyze = QPushButton("🔍 Автоанализ")
        self.btn_plot_dist = QPushButton("📊 Распределение")
        btn_panel.addWidget(self.btn_analyze)
        btn_panel.addWidget(self.btn_plot_dist)
        
        # Графическая область
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        # Отчёт
        self.analysis_report = QTextEdit()
        self.analysis_report.setReadOnly(True)
        
        layout.addLayout(btn_panel)
        layout.addWidget(self.canvas)
        layout.addWidget(self.analysis_report)
        self.tab_analyze.setLayout(layout)
        self.tabs.addTab(self.tab_analyze, "Анализ")

    def init_clean_tab(self):
        self.tab_clean = QWidget()
        self.btn_clean = QPushButton("Удалить дубликаты")
        clean_layout = QVBoxLayout()
        clean_layout.addWidget(self.btn_clean)
        clean_layout.addStretch()
        self.tab_clean.setLayout(clean_layout)
        self.tabs.addTab(self.tab_clean, "Очистка")

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
        self.tabs.addTab(self.tab_missing, "Пропуски")

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
        self.tabs.addTab(self.tab_outliers, "Выбросы")

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
        
        # Обновляем список столбцов при загрузке данных
        self.btn_load.clicked.connect(self.update_text_columns)

    def update_text_columns(self):
        """Обновляет список столбцов для обработки текста"""
        if self.current_data is not None:
            self.cb_text_column.clear()
            self.cb_text_column.addItems(self.current_data.columns.tolist())

    def load_text_file(self):
        """Загрузка текстового файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите текстовый файл", 
            "", 
            "Text Files (*.txt);;CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.current_text = f.read()
                elif file_path.endswith('.csv'):
                    self.current_text = pd.read_csv(file_path).to_string()
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.current_text = str(json.load(f))
                
                self.text_display.setPlainText(self.current_text)
                self.btn_save.setEnabled(True)
                self.log_message(f"Текст загружен из {file_path}")
            except Exception as e:
                self.log_message(f"Ошибка загрузки текста: {str(e)}", error=True)
                
    def process_text(self):
        """Обработка текста с возможностью произвольного порядка операций"""
        if not self.current_text:
            return
            
        try:
            self.show_progress(True)
            
            # Создаем процессор и добавляем выбранные операции
            processor = TextProcessor()
            
            # Добавляем операции в ЛЮБОМ порядке (примерная логика)
            if self.cb_remove_html.isChecked():
                processor.add_operation('remove_html')
                
            if self.cb_numbers_strategy.currentText() != "Оставить числа":
                strategy = 'remove' if self.cb_numbers_strategy.currentText() == "Удалить числа" else 'replace'
                processor.add_operation('handle_numbers', strategy=strategy)
                
            if self.cb_remove_special.isChecked():
                processor.add_operation('remove_special_chars')
                
            if self.cb_tokenize.currentText() != "Не токенизировать":
                token_type = 'word' if self.cb_tokenize.currentText() == "По словам" else 'sentence'
                processor.add_operation('tokenize', 
                                    token_type=token_type,
                                    lang=self.cb_language.currentText())
                
            if self.cb_remove_stopwords.isChecked():
                processor.add_operation('remove_stopwords',
                                    lang=self.cb_language.currentText())
                
            if self.cb_normalization.currentText() != "Не нормализовать":
                method = 'stem' if self.cb_normalization.currentText() == "Стемминг" else 'lemmatize'
                processor.add_operation('normalize',
                                    method=method,
                                    lang=self.cb_language.currentText())
            
            # Выполняем все операции
            processed_result = processor.execute(str(self.current_text))
            
            # Если результат - список (например, после токенизации), соединяем его в строку
            if isinstance(processed_result, list):
                processed_result = ' '.join(processed_result)
                
            self.current_text = processed_result
            self.text_display.setPlainText(self.current_text)
            self.log_message("Текст успешно обработан")
            
        except Exception as e:
            self.log_message(f"Ошибка обработки: {str(e)}", error=True)
        finally:
            self.show_progress(False)
    # Остальные методы остаются без изменений

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите файл данных", 
            "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.current_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.current_data = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    self.current_data = pd.read_json(file_path)
                
                self.display_data()
                self.btn_save.setEnabled(True)
                self.save_state()
                self.log_message(f"Данные загружены из {file_path}")
            except Exception as e:
                self.log_message(f"Ошибка загрузки: {str(e)}", error=True)

    def save_data(self):
        if self.current_data is not None or self.current_text is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Сохранить результат", 
                "", 
                "CSV Files (*.csv);;Text Files (*.txt);;JSON Files (*.json)"
            )
            if file_path:
                try:
                    if file_path.endswith('.csv') and self.current_data is not None:
                        self.current_data.to_csv(file_path, index=False)
                    elif file_path.endswith('.txt') and self.current_text is not None:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(self.current_text if isinstance(self.current_text, str) 
                                  else '\n'.join(self.current_text))
                    elif file_path.endswith('.json') and self.current_data is not None:
                        self.current_data.to_json(file_path, indent=4)
                    
                    self.log_message(f"Данные сохранены в {file_path}")
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
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())