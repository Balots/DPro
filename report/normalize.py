from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

class NormalizeData:
    def __init__(
        self,
        data: pd.DataFrame,
        columns: list = None,
        feature_range: tuple = (0, 1),
        source_name: str = None
    ):
        """
        🔧 Класс нормализации числовых данных с использованием MinMaxScaler

        Параметры:
            data: DataFrame с данными
            columns: список столбцов для нормализации (по умолчанию — все числовые)
            feature_range: целевой диапазон, например (0, 1)
            source_name: имя источника данных
        """
        self.data = data.copy()
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.feature_range = feature_range
        self.source_name = source_name or "неизвестный источник"
        self.before_stats = None
        self.after_stats = None
        self.result = None

    @staticmethod
    def from_file(file_path: str, columns: list = None, feature_range: tuple = (0, 1)) -> 'NormalizeData':
        """
        📁 Загружает данные из файла и возвращает экземпляр NormalizeData

        Поддерживаемые форматы: .csv, .xlsx, .json, .parquet
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"🚨 Файл не найден: {file_path}")

        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"❌ Неподдерживаемый формат файла: {ext}")

        return NormalizeData(df, columns=columns, feature_range=feature_range, source_name=path.name)

    def _select_numeric_columns(self) -> list:
        """🔎 Автоматический выбор числовых столбцов"""
        return self.data.select_dtypes(include=['number']).columns.tolist()

    def run(self) -> pd.DataFrame:
        """🚀 Запускает нормализацию данных"""
        df = self.data.copy()
        self.before_stats = df[self.columns].describe().T[['min', 'max']]

        scaler = MinMaxScaler(feature_range=self.feature_range)
        df[self.columns] = scaler.fit_transform(df[self.columns])

        self.after_stats = df[self.columns].describe().T[['min', 'max']]
        self.result = df
        return self.result

    def info(self) -> str:
        """📝 Возвращает текстовый отчет о нормализации"""
        info_str = (
            f"\n🔧 Нормализация данных из файла: '{self.source_name}'\n"
            f"📊 Столбцы для нормализации: {self.columns}\n"
            f"📐 Целевой диапазон: {self.feature_range}\n"
            f"{'═'*60}\n"
        )

        if self.before_stats is not None and self.after_stats is not None:
            info_str += "📉 Диапазоны значений ДО и ПОСЛЕ:\n"
            for col in self.columns:
                before_min = self.before_stats.loc[col, 'min']
                before_max = self.before_stats.loc[col, 'max']
                after_min = self.after_stats.loc[col, 'min']
                after_max = self.after_stats.loc[col, 'max']
                info_str += (f"  ▪️ {col}: "
                             f"до [{before_min:.2f}, {before_max:.2f}] → "
                             f"после [{after_min:.2f}, {after_max:.2f}]\n")
            info_str += f"{'═'*60}\n"
        else:
            info_str += "ℹ️ Запустите .run() для отображения статистики до/после.\n"

        return info_str

    def get_answ(self) -> pd.DataFrame:
        """📤 Возвращает нормализованные данные"""
        if self.result is None:
            self.run()
        return self.result
