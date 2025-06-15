import App


"""
If you wished to use App interface to analyse your data instead of Method. Just run file. 
"""

import sys
from PyQt5.QtWidgets import QApplication
from App.MainWindow import DataProcessingApp

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Создаем QApplication первым
    
    # Настройка стиля (опционально)
    app.setStyle('Fusion')
    
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())