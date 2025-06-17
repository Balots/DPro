import sys
from PyQt5.QtWidgets import QApplication
from App.MainWindow import DataProcessingApp  # Импортируем из правильного модуля

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Создаем QApplication
    
    # Настройка стиля (как в MainWindow.py)
    app.setStyle('Fusion')
    window = DataProcessingApp()
    window.show()
    
    sys.exit(app.exec_())  # Запускаем цикл событий