from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

class StandardizeData:
    def __init__(self, data: pd.DataFrame, columns: list = None, source_name: str = None):
        """
        ⚙️ Класс стандартизации числовых данных (среднее = 0, std = 1)

        Параметры:
            data: DataFrame с данными
            columns: список столбцов для стандартизации (по умолчанию — все числовые)
            source_name: имя источника данных
        """
        self.data = data.copy()
        self.columns = columns if columns is not None else self._select_numeric_columns()
        self.source_name = source_name or "неизвестный источник"
        self.before_stats = None
        self.after_stats = None
        self.result = None

    @staticmethod
    def from_file(file_path: str, columns: list = None) -> 'StandardizeData':
        """
        📁 Загружает данные из файла и возвращает экземпляр StandardizeData

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

        return StandardizeData(df, columns=columns, source_name=path.name)

    def run(self) -> pd.DataFrame:
        """🚀 Запускает стандартизацию"""
        df = self.data.copy()
        self.before_stats = df[self.columns].describe().T[['mean', 'std']]

        scaler = StandardScaler()
        df[self.columns] = scaler.fit_transform(df[self.columns])

        self.after_stats = df[self.columns].describe().T[['mean', 'std']]
        self.result = df
        return self.result

    def info(self) -> str:
        """📋 Возвращает отчет по результатам стандартизации"""
        info_str = (
            f"\n⚙️ Стандартизация данных из файла: '{self.source_name}'\n"
            f"📊 Столбцы для стандартизации: {self.columns}\n"
            f"🎯 Цель: привести значения к среднему 0 и стандартному отклонению 1\n"
            f"{'═'*60}\n"
        )

        if self.before_stats is not None and self.after_stats is not None:
            info_str += "📉 Средние и стандартные отклонения ДО и ПОСЛЕ:\n"
            for col in self.columns:
                before_mean = self.before_stats.loc[col, 'mean']
                before_std = self.before_stats.loc[col, 'std']
                after_mean = self.after_stats.loc[col, 'mean']
                after_std = self.after_stats.loc[col, 'std']
                info_str += (f"  ▪️ {col}: "
                             f"до [μ={before_mean:.2f}, σ={before_std:.2f}] → "
                             f"после [μ={after_mean:.2f}, σ={after_std:.2f}]\n")
            info_str += f"{'═'*60}\n"
        else:
            info_str += "ℹ️ Запустите .run() для отображения статистики до/после.\n"

        return info_str

    def get_answ(self) -> pd.DataFrame:
        """📤 Возвращает стандартизированные данные"""
        if self.result is None:
            self.run()
        return self.result

    def _select_numeric_columns(self) -> list:
        """🔎 Автоматически выбирает числовые столбцы"""
        return self.data.select_dtypes(include=['number']).columns.tolist()
