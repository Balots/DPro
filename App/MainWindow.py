import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit, 
                             QComboBox, QLineEdit, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt
from DataProcessing import CleanData
from DataProcessing import HandleMissingValues
from DataProcessing import DetectAndRemoveOutliers
from DataProcessing import NormalizeData, StandardizeData


class DataProcessingApp(QMainWindow):

    __APP__ = QApplication(sys.argv)


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing Tool")
        self.setGeometry(100, 100, 1000, 700)
        self.current_data = None
        self.init_ui()

    def exec(self):
        sys.exit(self.__APP__.exec_())

    def init_ui(self):
        # Главный виджет и layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Панель управления
        control_panel = QHBoxLayout()
        
        # Кнопки загрузки/сохранения
        self.btn_load = QPushButton("Загрузить данные")
        self.btn_load.clicked.connect(self.load_data)
        self.btn_save = QPushButton("Сохранить результат")
        self.btn_save.clicked.connect(self.save_data)
        self.btn_save.setEnabled(False)
        
        control_panel.addWidget(self.btn_load)
        control_panel.addWidget(self.btn_save)
        
        # Табы для разных операций
        self.tabs = QTabWidget()
        
        # Вкладка просмотра данных
        self.tab_data = QWidget()
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        data_layout = QVBoxLayout()
        data_layout.addWidget(self.data_table)
        self.tab_data.setLayout(data_layout)
        self.tabs.addTab(self.tab_data, "Данные")
        
        # Вкладка очистки данных
        self.tab_clean = QWidget()
        self.btn_clean = QPushButton("Удалить дубликаты")
        self.btn_clean.clicked.connect(self.clean_data)
        clean_layout = QVBoxLayout()
        clean_layout.addWidget(self.btn_clean)
        clean_layout.addStretch()
        self.tab_clean.setLayout(clean_layout)
        self.tabs.addTab(self.tab_clean, "Очистка")
        
        # Вкладка обработки пропусков
        self.tab_missing = QWidget()
        self.cb_num_strategy = QComboBox()
        self.cb_num_strategy.addItems(["mean", "median"])
        self.cb_cat_strategy = QComboBox()
        self.cb_cat_strategy.addItems(["mode", "constant"])
        self.btn_process_missing = QPushButton("Обработать пропуски")
        self.btn_process_missing.clicked.connect(self.process_missing)
        
        missing_layout = QVBoxLayout()
        missing_layout.addWidget(QLabel("Стратегия для чисел:"))
        missing_layout.addWidget(self.cb_num_strategy)
        missing_layout.addWidget(QLabel("Стратегия для категорий:"))
        missing_layout.addWidget(self.cb_cat_strategy)
        missing_layout.addWidget(self.btn_process_missing)
        missing_layout.addStretch()
        self.tab_missing.setLayout(missing_layout)
        self.tabs.addTab(self.tab_missing, "Пропуски")
        
        # Вкладка выбросов
        self.tab_outliers = QWidget()
        self.le_outlier_cols = QLineEdit()
        self.le_outlier_cols.setPlaceholderText("Укажите столбцы через запятую")
        self.btn_remove_outliers = QPushButton("Удалить выбросы")
        self.btn_remove_outliers.clicked.connect(self.remove_outliers)
        
        outliers_layout = QVBoxLayout()
        outliers_layout.addWidget(QLabel("Столбцы для обработки:"))
        outliers_layout.addWidget(self.le_outlier_cols)
        outliers_layout.addWidget(self.btn_remove_outliers)
        outliers_layout.addStretch()
        self.tab_outliers.setLayout(outliers_layout)
        self.tabs.addTab(self.tab_outliers, "Выбросы")
        
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
    
    def load_data(self):
        """Загрузка данных из CSV-файла"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл данных", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.current_data = pd.read_csv(file_path)
                self.display_data()
                self.btn_save.setEnabled(True)
                self.log_message(f"Данные загружены из {file_path}")
            except Exception as e:
                self.log_message(f"Ошибка загрузки: {str(e)}", error=True)
    
    def save_data(self):
        """Сохранение обработанных данных"""
        if self.current_data is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", "", "CSV Files (*.csv)")
            if file_path:
                try:
                    self.current_data.to_csv(file_path, index=False)
                    self.log_message(f"Данные сохранены в {file_path}")
                except Exception as e:
                    self.log_message(f"Ошибка сохранения: {str(e)}", error=True)
    
    def display_data(self):
        """Отображение данных в таблице"""
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
        """Очистка данных от дубликатов"""
        if self.current_data is not None:
            try:
                processor = CleanData(self.current_data)
                self.current_data = processor.run()
                self.display_data()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка очистки: {str(e)}", error=True)
    
    def process_missing(self):
        """Обработка пропущенных значений"""
        if self.current_data is not None:
            try:
                processor = HandleMissingValues(
                    self.current_data,
                    numeric_strategy=self.cb_num_strategy.currentText(),
                    categorical_strategy=self.cb_cat_strategy.currentText()
                )
                self.current_data = processor.run()
                self.display_data()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка обработки пропусков: {str(e)}", error=True)
    
    def remove_outliers(self):
        """Удаление выбросов"""
        if self.current_data is not None:
            try:
                columns = [col.strip() for col in self.le_outlier_cols.text().split(",")] if self.le_outlier_cols.text() else None
                processor = DetectAndRemoveOutliers(self.current_data, columns=columns)
                self.current_data = processor.run()
                self.display_data()
                self.log_message(processor.info())
            except Exception as e:
                self.log_message(f"Ошибка удаления выбросов: {str(e)}", error=True)
    
    def log_message(self, message, error=False):
        """Логирование сообщений"""
        if error:
            self.log.append(f"<font color='red'>{message}</font>")
        else:
            self.log.append(message)

if __name__ == "__main__":
    print('This is not a lib')

