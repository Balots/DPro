import pandas as pd
from pathlib import Path

def check_duplicates_file(file_path: str, show_report: bool = True) -> dict:
    """
    🔍 Анализирует файл (CSV, XLSX, JSON, Parquet) на наличие дубликатов

    Параметры:
        file_path: Путь к файлу
        show_report: Показывать ли красивый отчет (по умолчанию True)

    Возвращает:
        Словарь с результатами:
        {
            "total_rows": Общее количество строк,
            "duplicates_count": Найденные дубликаты,
            "removed_rows": Удаленные строки,
            "remaining_rows": Оставшиеся строки,
            "has_duplicates": Остались ли дубликаты
        }

    Исключения:
        FileNotFoundError: Если файл не существует
        ValueError: Неподдерживаемый формат файла
        pd.errors.EmptyDataError: Если файл пустой
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"🚨 Файл не найден: {file_path}")

        # Определение формата и чтение файла
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

        if df.empty:
            print("⚠️ Внимание: Файл пустой!")
            return {
                "total_rows": 0,
                "duplicates_count": 0,
                "removed_rows": 0,
                "remaining_rows": 0,
                "has_duplicates": False
            }

        # Анализ дубликатов
        duplicates = df[df.duplicated()]
        dup_count = len(duplicates)
        cleaned_df = df.drop_duplicates()
        removed = len(df) - len(cleaned_df)

        report = {
            "total_rows": len(df),
            "duplicates_count": dup_count,
            "removed_rows": removed,
            "remaining_rows": len(cleaned_df),
            "has_duplicates": cleaned_df.duplicated().any()
        }

        # Красивый отчет
        if show_report:
            print("\n" + "═"*50)
            print("🔍 АНАЛИЗ ДУБЛИКАТОВ".center(50))
            print("═"*50)
            print(f"📁 Файл: {path.name}")
            print(f"📊 Всего строк: {report['total_rows']:>30}")
            print(f"🔍 Найдено дубликатов: {dup_count:>23}")
            print(f"🧹 Удалено строк: {removed:>28}")
            print(f"✅ Уникальных записей: {report['remaining_rows']:>22}")

            if dup_count > 0:
                print("\n🔎 Примеры дубликатов:")
                print(duplicates.head(2).to_string(index=False))

            print("\n" + "═"*50)
            if dup_count == 0:
                print("🎉 Отлично! Дубликатов не обнаружено!")
            else:
                print(f"💾 Дубликаты удалены")
            print("═"*50)

        return report

    except pd.errors.EmptyDataError:
        print("🚨 Ошибка: Файл поврежден или не содержит данных")
        raise
    except Exception as e:
        print(f"🚨 Неожиданная ошибка: {str(e)}")
        raise
