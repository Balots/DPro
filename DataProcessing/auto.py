import pandas as pd
from .base import DataProcessing
from DataProcessing import *

class AutoAnal(DataProcessing):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.result = None

    def auto_analyze(self):
        """
        Пошаговый анализ и автоматическая обработка данных:
        удаление дубликатов, обработка пропущенных значений, удаление выбросов и нормализация.
        Возвращает очищенный и нормализованный датафрейм.
        """
        data = self.data
        if data.duplicated().any():
            cleaner = CleanData(data)  # создаём объект очистки дубликатов
            print("Рекомендация: удалить дубликаты")
            print(cleaner.info())  # выводим информацию о действии
            data = cleaner.run()  # выполняем удаление
        else:
            print("Дубликаты не найдены — пропускаем очистку")

        if data.isnull().any().any():
            # Используем стратегию: числовые — KNN, категориальные — мода (наиболее частое значение)
            missing_handler = HandleMissingValues(
                data,
                numeric_strategy='knn',
                categorical_strategy='mode'
            )
            print("Рекомендация: обработать пропущенные значения")
            print(missing_handler.info())  # выводим описание действия
            try:
                data = missing_handler.run()  # применяем замену пропущенных значений
            except Exception:
                pass
        else:
            print("Пропущенных значений нет — пропускаем обработку пропусков")

        # --- 3. Выявление и удаление выбросов ---
        def has_outliers(df, columns=None, factor=1.5):
            """
            Проверка наличия выбросов с использованием IQR-метода (ручная проверка).
            Возвращает True, если хотя бы в одном столбце есть выбросы.
            """
            columns = columns or df.select_dtypes(include='number').columns.tolist()
            for col in columns:
                Q1 = df[col].quantile(0.25)  # первый квартиль
                Q3 = df[col].quantile(0.75)  # третий квартиль
                IQR = Q3 - Q1  # межквартильный размах
                lower_bound = Q1 - factor * IQR  # нижняя граница
                upper_bound = Q3 + factor * IQR  # верхняя граница
                # Если есть значения за пределами допустимого диапазона — выбросы есть
                if df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0] > 0:
                    return True
            return False

        # Если найдены выбросы, запускаем их удаление
        if has_outliers(data):
            # Используем автоматический выбор метода на основе минимальной асимметрии
            outlier_handler = DetectAndRemoveOutliers(data, method='auto')
            print("Рекомендация: удалить выбросы (автоматически подобранный метод)")
            print(outlier_handler.info())  # отображаем, что будет сделано
            try:
                data = outlier_handler.run()  # запускаем удаление выбросов
            except Exception:
                pass
        else:
            print("Выбросов не найдено — пропускаем удаление выбросов")

        # --- 4. Нормализация числовых данных ---
        numeric_cols = data.select_dtypes(include='number').columns.tolist()

        if numeric_cols:
            # Считаем минимумы и максимумы всех числовых столбцов
            ranges = data[numeric_cols].agg(['min', 'max'])
            # Проверяем, есть ли значения с большим диапазоном — если да, нормализуем
            need_normalize = any((ranges.loc['max'] - ranges.loc['min']) > 1)

            if need_normalize:
                normalizer = NormalizeData(data)  # создаём объект нормализации
                print("Рекомендация: нормализовать числовые данные")
                print(normalizer.info())  # выводим описание действия
                data = normalizer.run()  # выполняем нормализацию
            else:
                print("Нормализация не требуется (диапазоны значений уже в порядке)")
        else:
            print("Нет числовых столбцов — пропускаем нормализацию")

        # Возвращаем финальный обработанный датафрейм
        return data
    def run(self):
        self.result = self.auto_analyze()

    def info(self) -> str:
        return "info"

    def get_answ(self) -> pd.DataFrame:
        if self.result is None:
            self.run()
        return self.result