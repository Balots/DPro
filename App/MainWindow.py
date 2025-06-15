import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit,
    QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QStatusBar, QProgressBar, QAction, QToolBar, QButtonGroup, QRadioButton
)
from PyQt5.QtGui import QPalette, QColor, QTextDocument, QDoubleValidator
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Detector.Detector import Detector
from DataProcessing import CleanData, HandleMissingValues, DetectAndRemoveOutliers, NormalizeData, StandardizeData
import matplotlib.pyplot as plt


class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.current_data = None
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
        self.btn_save = QPushButton("Сохранить результат")
        self.btn_save.setEnabled(False)
        control_panel.addWidget(self.btn_load)
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
        
        # Вкладка масштабирования
        self.init_scaling_tab()

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
        self.btn_clean.clicked.connect(self.clean_data)
        self.btn_process_missing.clicked.connect(self.process_missing)
        self.btn_remove_outliers.clicked.connect(self.remove_outliers)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_plot_dist.clicked.connect(self.plot_distribution)
        self.btn_apply_scaling.clicked.connect(self.apply_scaling)

    # Основные методы обработки данных
    def load_data(self):
        """Загрузка данных с поддержкой CSV, Excel, JSON и Parquet"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл данных",
            "",
            "Все поддерживаемые (*.csv *.xlsx *.json *.parquet);;"
            "CSV (*.csv);;Excel (*.xlsx);;JSON (*.json);;Parquet (*.parquet)"
        )
        
        if not file_path:
            return

        try:
            self.show_progress(True)
            
            # Определяем формат по расширению файла
            if file_path.endswith('.csv'):
                self.current_data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.current_data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.current_data = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                self.current_data = pd.read_parquet(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")

            self.display_data()
            self.btn_save.setEnabled(True)
            self.save_state()
            self.log_message(f"Данные загружены из {file_path}")
            
        except Exception as e:
            self.log_message(f"Ошибка загрузки: {str(e)}", error=True)
        finally:
            self.show_progress(False)

    def save_data(self):
        """Сохранение данных в различных форматах"""
        if self.current_data is None:
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;JSON (*.json);;Parquet (*.parquet)"
        )
        
        if not file_path:
            return

        try:
            self.show_progress(True)
            
            # Добавляем правильное расширение, если его нет
            if selected_filter == "CSV (*.csv)" and not file_path.endswith('.csv'):
                file_path += '.csv'
            elif selected_filter == "Excel (*.xlsx)" and not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            elif selected_filter == "JSON (*.json)" and not file_path.endswith('.json'):
                file_path += '.json'
            elif selected_filter == "Parquet (*.parquet)" and not file_path.endswith('.parquet'):
                file_path += '.parquet'

            # Сохранение в соответствующем формате
            if file_path.endswith('.csv'):
                self.current_data.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                self.current_data.to_excel(file_path, index=False)
            elif file_path.endswith('.json'):
                self.current_data.to_json(file_path, orient='records')
            elif file_path.endswith('.parquet'):
                self.current_data.to_parquet(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")

            self.log_message(f"Данные сохранены в {file_path}")
            
        except Exception as e:
            self.log_message(f"Ошибка сохранения: {str(e)}", error=True)
        finally:
            self.show_progress(False)

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
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())
    