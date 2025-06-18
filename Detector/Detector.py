import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import json

class Detector:
    def __init__(self, check_abnormal:bool, check_missing:bool, check_duplicates:bool, check_scaling:bool, hampel_threshold:float = 3.0,
                 iqr_multiplier:float = 1.5, skewness_threshold:float = 2.0, kurtosis_threshold:float = 3.5):
        self.check_abnormal = check_abnormal
        self.check_missing = check_missing
        self.check_duplicates = check_duplicates
        self.check_scaling = check_scaling
        self.hampel_threshold = hampel_threshold
        self.iqr_multiplier = iqr_multiplier
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold


    def check_dataframe(self, filename, is_df=False):

        '''Проверка на наличие пропущенные значений, дубликатов, выбросов и рекомендации по нормализации/
        стандартизации данных в столбцах'''

        df = filename if is_df else pd.read_csv(filename)
        profile = ProfileReport(df, title="to check")

        outcome = {'Overall alerts/Общие проблемы': json.loads(profile.to_json())['alerts'],
                   'Missing values/Пропущенные значения': self.find_missing(profile),
                   'Duplicate values/Дубликаты значений ': self.find_duplicates(profile)}

        abnormal = self.find_abnormal(profile, df, self.hampel_threshold, self.iqr_multiplier, self.skewness_threshold, self.kurtosis_threshold)
        scaling = self.recommend_scaling_methods(df, profile)
#        for col, data in abnormal.items():
#            print(f'результат для колонки {col}')

#            for method in ['IQR', 'Modified_Z_score', 'Skewness_Method', 'Percentile', 'Kurtosis_Method']:
#                m = data[method]
#                if m['count'] != 0:
#                    print(f"\n{method.replace('_', ' ')}:")
#                    print(f"Метод: {m['method']}")
#                    if 'threshold' in m:
#                        print(f" Порог: {m['threshold']}")
#                    if 'direction' in m:
#                        print(f" Направление: {m['direction']}")
#                    print(f" Найдено выбросов: {m['count']} ({m['count'] / len(df) * 100:.1f}%)")

#        recommendations = self.recommend_scaling_methods(df, profile)
#        print(recommendations)
        return outcome, abnormal, scaling

    def increase_threshold(self, increasing_multiplier:int):
        self.hampel_threshold += 0.2 * increasing_multiplier
        self.iqr_multiplier += 0.1 * increasing_multiplier
        self.skewness_threshold += 0.14 * increasing_multiplier
        self.kurtosis_threshold += 0.22 * increasing_multiplier


    def decrease_threshold(self, decreasing_multiplier:int):
        self.hampel_threshold -= 0.2 * decreasing_multiplier
        self.iqr_multiplier -= 0.1 * decreasing_multiplier
        self.skewness_threshold -= 0.14 * decreasing_multiplier
        self.kurtosis_threshold -= 0.22 * decreasing_multiplier


    def find_missing(self, profile):
        missing_report = json.loads(profile.to_json())['table']['n_cells_missing']
        if missing_report != 0:
            report = 'Missing values exist/Пропущенные значения присутствуют'
            percentage = json.loads(profile.to_json())['table']['p_cells_missing']
            report += f', Percentage/Процент пропущенных значений:{percentage}'
        else:
            report = 'No missing values found/Пропущенные значения отсутствуют'
        return report


    def find_duplicates(self, profile):
        duplicates_report = json.loads(profile.to_json())['table']['n_duplicates']
        if duplicates_report != 0:
            report = 'Duplicates exist/Дубликаты присутствуют'
            percentage = json.loads(profile.to_json())['table']['p_duplicates']
            report += f', Percentage/Процент дубликатов:{percentage}'
        else:
            report = 'No duplicates found/Дубликаты отсутствуют'
        return report


    def find_abnormal(self, profile, df, hampel_threshold,
                    iqr_multiplier,
                    skewness_threshold,
                    kurtosis_threshold):
        """
            Анализирует выбросы с помощью 5 различных методов:
            1. IQR метод (межквартильный размах)
            2. Модифицированный Z-score (фильтр Хемпеля)
            3. 5й и 95й процентили
            4. Метод на основе коэффициента асимметрии
            5. Метод на основе эксцесса

            Параметры:
            profile - отчет ydata
            df - исходный DataFrame
            hampel_threshold - порог для фильтра Хемпеля
            iqr_multiplier - множитель для IQR метода
            std_dev_multiplier - множитель для стандартного отклонения
            skewness_threshold - порог для метода асимметрии
            kurtosis_threshold - порог для метода эксцесса
            """
        report = json.loads(profile.to_json())
        result = {}

        for column_name in report['variables']:
            if report['variables'][column_name]['type'] == 'Numeric':
                stats = report['variables'][column_name]
                col_data = df[column_name].dropna()

                # 1. IQR метод
                q1 = stats['25%']
                q3 = stats['75%']
                iqr = stats['iqr']
                lower_iqr = q1 - iqr_multiplier * iqr
                upper_iqr = q3 + iqr_multiplier * iqr
                iqr_outliers = col_data[(col_data < lower_iqr) | (col_data > upper_iqr)]

                # 2. Модифицированный Z-score (фильтр Хемпеля)
                median = stats['50%']
                mad = stats['mad']
                modified_z = 0.6745 * (col_data - median) / mad
                hampel_outliers = col_data[np.abs(modified_z) > hampel_threshold]

                # 3. Процентили (P5-P95)
                p5, p95 = stats['5%'], stats['95%']
                percentile_outliers = col_data[(col_data < p5) | (col_data > p95)]

                # 4. Метод на основе коэффициента асимметрии
                skewness = stats['skewness']
                if abs(skewness) > skewness_threshold:
                    skewness_dir = 'right' if skewness > 0 else 'left'
                    skewness_outliers = col_data[
                        (col_data > stats['mean'] + 2 * stats['std']) if skewness > 0
                        else (col_data < stats['mean'] - 2 * stats['std'])
                    ]
                else:
                    skewness_outliers = pd.Series(dtype=float)

                # 5. Метод на основе эксцесса
                kurtosis = stats['kurtosis']

                if kurtosis > kurtosis_threshold:
                    kurtosis_outliers = col_data[np.abs((col_data - median) / mad) > 3.5]
                else:
                    kurtosis_outliers = pd.Series(dtype=float)

                # Сохраняем результаты
                result[column_name] = {
                    'IQR': {
                        'method': f'IQR ({iqr_multiplier}×)',
                        'outliers': iqr_outliers.to_list(),
                        'count': len(iqr_outliers)
                    },
                    'Modified_Z_score': {
                        'method': 'Hampel (MAD-based)',
                        'threshold': hampel_threshold,
                        'outliers': hampel_outliers.to_list(),
                        'count': len(hampel_outliers)
                    },
                    'Percentile': {'method': 'percentile', 'bounds': [p5, p95],
                                   'outliers': percentile_outliers.tolist(),
                                   'count': len(percentile_outliers)},
                    'Skewness_Method': {
                        'method': f'Skewness (>{skewness_threshold})',
                        'threshold': skewness_threshold,
                        'direction': 'right' if skewness > 0 else 'left',
                        'outliers': skewness_outliers.to_list(),
                        'count': len(skewness_outliers)
                    },
                    'Kurtosis_Method': {
                        'method': f'Kurtosis (>{kurtosis_threshold})',
                        'threshold': kurtosis_threshold,
                        'outliers': kurtosis_outliers.to_list(),
                        'count': len(kurtosis_outliers)
                    }
                }
        return result


    def recommend_scaling_methods(self, df, profile):
        """
        Рекомендует метод масштабирования (нормализацию или стандартизацию) для каждого числового столбца

        Параметры:
        df - исходный DataFrame
        profile - отчет ydata-profiling

        Возвращает:
        Словарь с рекомендациями для каждого столбца
        """
        report = json.loads(profile.to_json())
        recommendations = {}

        for column_name in report['variables']:
            if report['variables'][column_name]['type'] == 'Numeric':
                stats = report['variables'][column_name]
                col_data = df[column_name].dropna()

                # 1. проверка на выбросы с помощью IQR
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = stats['iqr']
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                has_outliers = (stats['min'] < lower_bound) or (stats['max'] > upper_bound)

                # 2. проверка симметричности распределения
                mean_median_diff = abs(stats['mean'] - stats['50%'])
                is_symmetric = mean_median_diff < (0.1 * stats['mean'])

                # 3. проверка коэффициента вариации
                cv = stats['cv']
                is_high_variability = cv > 0.5

                # 4. анализ формы распределения
                skewness = abs(stats['skewness'])
                is_highly_skewed = skewness > 1

                # 5. правила принятия решения
                reasons = []
                if has_outliers:
                    reasons.append("наличие выбросов")
                if not is_symmetric:
                    reasons.append("асимметричное распределение")
                if is_high_variability:
                    reasons.append("высокая вариативность")
                if is_highly_skewed:
                    reasons.append("сильная скошенность")

                if has_outliers or not is_symmetric or is_high_variability or is_highly_skewed:
                    recommendation = "стандартизировать"
                else:
                    recommendation = "нормализовать"

                recommendations[column_name] = {
                    'Рекомендация': recommendation,
                    'причина': reasons if reasons else ["равномерное распределение без выбросов"]
                }

        return recommendations


class IDet:
    def __init__(self, check_abnormal: bool, check_missing: bool, check_duplicates: bool, check_scaling: bool,  *args):
        self.standart_settings = args
        self.detector = Detector(check_abnormal, check_missing, check_duplicates, check_scaling, *self.standart_settings)

    def logging_results(self, filename):
        return self.detector.check_dataframe(filename)


if __name__ == '__main__':
    print(IDet(True, True, True, True).logging_results('Accidents.csv'))