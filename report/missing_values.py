import pandas as pd
import numpy as np
from pathlib import Path

class MissingValuesAnalyzer:
    def __init__(self, data: pd.DataFrame, source_name: str = "DataFrame"):
        """🔍 Анализатор пропущенных значений с визуализацией"""
        self.original_data = data.copy()
        self.source_name = source_name
        self.processed_data = None
        self.missing_stats = {}

    @staticmethod
    def from_file(file_path: str) -> 'MissingValuesAnalyzer':
        """
        📁 Загружает данные из файла и создает экземпляр анализатора

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

        return MissingValuesAnalyzer(df, source_name=path.name)

    def analyze(self) -> dict:
        """📊 Анализирует пропущенные значения"""
        stats = {}
        for col in self.original_data.columns:
            missing = self.original_data[col].isnull().sum()
            if missing > 0:
                stats[col] = {
                    'dtype': str(self.original_data[col].dtype),
                    'missing': missing,
                    'pct': round(missing / len(self.original_data) * 100, 2),
                    'suggested_strategy': self._suggest_strategy(col)
                }
        self.missing_stats = stats
        return stats

    def _suggest_strategy(self, col: str) -> str:
        """🤖 Предлагает стратегию обработки"""
        if pd.api.types.is_numeric_dtype(self.original_data[col]):
            return 'median' if self.original_data[col].skew() > 1 else 'mean'
        else:
            return 'mode' if len(self.original_data[col].unique()) < 20 else 'constant'

    def visualize_missing(self):
        """📉 Визуализирует распределение пропусков"""
        if not self.missing_stats:
            print("🎉 Пропущенных значений не обнаружено!")
            return

        print("\n" + "═" * 60)
        print("📊 РАСПРЕДЕЛЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ".center(60))
        print(f"📁 Источник: {self.source_name}".center(60))
        print("═" * 60)

        for col, stat in self.missing_stats.items():
            print(f"\n📌 Столбец: {col} ({stat['dtype']})")
            print(f"   ▪️ Пропусков: {stat['missing']} ({stat['pct']}%)")
            print(f"   💡 Рекомендуемая стратегия: {stat['suggested_strategy']}")

            # Визуализация для числовых данных
            if pd.api.types.is_numeric_dtype(self.original_data[col]):
                desc = self.original_data[col].describe()
                print(f"   📐 Статистика: min={desc['min']:.1f}, 25%={desc['25%']:.1f}, "
                      f"median={desc['50%']:.1f}, 75%={desc['75%']:.1f}, max={desc['max']:.1f}")

        print("\n" + "═" * 60)

    def process(self, strategies: dict = None) -> pd.DataFrame:
        """🛠️ Обрабатывает пропуски согласно стратегиям"""
        if not self.missing_stats:
            self.analyze()

        strategies = strategies or {
            'numeric': 'median',
            'categorical': 'mode'
        }

        df = self.original_data.copy()

        print(f"\n🛠️ НАЧАЛО ОБРАБОТКИ ПРОПУСКОВ ДЛЯ: {self.source_name}")
        for col in self.missing_stats.keys():
            strategy = self._get_strategy_for_column(col, strategies)
            df[col] = self._apply_strategy(df[col], strategy)
            print(f"   ✔ {col}: {self._strategy_description(strategy)}")

        self.processed_data = df
        print("\n✅ ОБРАБОТКА ЗАВЕРШЕНА!")
        self._verify_processing()
        return df

    def _get_strategy_for_column(self, col: str, strategies: dict) -> str:
        """📌 Выбирает стратегию для столбца"""
        if col in strategies:
            return strategies[col]
        return strategies['numeric' if pd.api.types.is_numeric_dtype(self.original_data[col]) else 'categorical']

    def _apply_strategy(self, series: pd.Series, strategy: str):
        """🔧 Применяет стратегию обработки"""
        if strategy == 'mean':
            return series.fillna(series.mean())
        elif strategy == 'median':
            return series.fillna(series.median())
        elif strategy == 'mode':
            return series.fillna(series.mode()[0])
        elif strategy == 'constant':
            return series.fillna('UNKNOWN' if series.dtype == 'object' else 0)
        elif strategy == 'drop':
            return series.dropna()
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

    def _strategy_description(self, strategy: str) -> str:
        """📝 Описание стратегии"""
        descriptions = {
            'mean': 'заполнение средним',
            'median': 'заполнение медианой',
            'mode': 'заполнение модой',
            'constant': 'заполнение константой',
            'drop': 'удаление строк'
        }
        return descriptions.get(strategy, strategy)

    def _verify_processing(self):
        """🔍 Проверяет результат обработки"""
        total_before = self.original_data.isnull().sum().sum()
        total_after = self.processed_data.isnull().sum().sum()
        affected_columns = len(self.missing_stats)

        if total_after == 0:
            print("🎉 Все пропуски успешно обработаны!")
        else:
            print(f"⚠️ Осталось {total_after} пропущенных значений из {total_before}")

        print("\n" + "═" * 60)
        print("🧮 СТАТИСТИКА ОБРАБОТКИ".center(60))
        print("═" * 60)
        print(f"📁 Источник: {self.source_name}")
        print(f"🧱 Строк: {len(self.original_data)} | Столбцов с пропусками: {affected_columns}")
        print(f"❌ Пропусков ДО: {total_before}")
        print(f"✅ Пропусков ПОСЛЕ: {total_after}")
        print("═" * 60)
