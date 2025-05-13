import numpy as np
import pandas as pd
import os 
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def try_read_table(table_path: str):
    if table_path.endswith("csv"):
        return pd.read_csv(table_path, on_bad_lines='skip', low_memory=False)
    elif table_path.endswith("xlsx"):
        return pd.read_excel(table_path)
    elif table_path.endswith("ods"):
        return pd.read_excel(table_path, engine='odf')

def prepare_data(dir_path: str):
    dfs = []
    dfs_names = []

    for table in os.listdir(dir_path):
            df = None

            try:
                df = try_read_table(table)
            except:
                print(table + " - not parsed\n")

            if df is not None:
                dfs.append(df)
                dfs_names.append(table)

    df = dfs[26].copy()
    # Create sequence number for each email AND НЭ combination
    df['seq_num'] = df.groupby(['Адрес электронной почты', 'НЭ']).cumcount() + 1

    # Define columns to pivot
    pivot_columns = {
        'ОЦЕНКА': 'ОЦЕНКА_',
        'Год': 'Год_',
        'Курс': 'Курс_',
        'Онлайн-курс': 'Онлайн-курс_',
        'ИУС': 'ИУС_',
        'Демо-версия': 'Демо-версия_',
        'Сдал-не-сдал': 'Сдал-не-сдал_'
    }

    # Create pivot tables with multi-index
    pivot_tables = {
        prefix: df.pivot_table(
            index=['Адрес электронной почты', 'НЭ'],
            columns='seq_num',
            values=col,
            aggfunc='first'  # In case of duplicates, take the first value
        ).add_prefix(prefix)
        for col, prefix in pivot_columns.items()
    }

    # Get email and НЭ metadata (first occurrence for each combination)
    email_metadata = df.groupby(['Адрес электронной почты', 'НЭ']).first()

    # Combine all pivot tables
    result_df = pd.concat([email_metadata] + list(pivot_tables.values()), axis=1)

    columns_to_drop = [
        'Группа', 'Курс', 'ОЦЕНКА', 'Год', 'Онлайн-курс', 'ИУС', 
        'Демо-версия', 'Сдал-не-сдал', 'Сдал/не сдал',
        'Больше половины онлайн-курса', 'Чек', 'Оценка за онлайн курс'
    ]

    result_df = (result_df
        .fillna('')
        .reset_index()
        .drop(columns=columns_to_drop)
    )

    base_df = result_df.copy()
    base_df.rename(columns={"Адрес электронной почты": "email"}, inplace=True)

    ne_encoded = pd.get_dummies(base_df['НЭ'], prefix='НЭ').astype(int)
    base_df = pd.concat([base_df, ne_encoded], axis=1)
    base_df.drop('НЭ', axis=1, inplace=True)

    dfs.pop(26)
    dfs_names.pop(26)
    
    path = "Учебник_Python Оценки-20241021_0653-semicolon_separated.csv"
    cur_df = pd.read_csv(dir_path + path, sep=';', encoding='utf-8', quoting=1, low_memory=False)
    dfs[14] = cur_df

    # dfs[1] - Выгрузка оценок общая 3 и 4 курсы.xlsx

    cur_df = dfs[1][["Почта", "Оценка", "Раздел", "ПериодОбучения", "Дисциплина"]]
    cur_df['Оценка'] = pd.to_numeric(cur_df['Оценка'], errors='coerce')

    def get_last_fail_position(group):
        # Sort by ПериодОбучения in ascending order
        sorted_group = group.sort_values('ПериодОбучения')
        # Find failing grades
        failing_grades = sorted_group['Оценка'] < 4
        if failing_grades.any():
            # Get the position (0-based) of the last failing grade
            return failing_grades.values[::-1].argmax()
        return 0  # Return 0 if no failing grades

    # Process Data Culture section
    dc_df = cur_df[cur_df["Раздел"] == "Data Culture"].copy()
    dc_latest = (dc_df.sort_values('ПериодОбучения', ascending=False)
                    .groupby('Почта')
                    .head(10)
                    .copy())

    dc_stats = dc_latest.groupby('Почта').agg({
        'Оценка': [
            ('dc_avg_grade', 'mean'),
            ('dc_grades_above_4', lambda x: (x >= 4).sum()),
            ('dc_grades_below_4', lambda x: (x < 4).sum()),
        ]
    })

    dc_last_fail = dc_latest.groupby('Почта').apply(get_last_fail_position)
    dc_stats[('Оценка', 'dc_last_fail_position')] = dc_last_fail

    # Process Major section
    major_df = cur_df[cur_df["Раздел"] == "Major"].copy()
    major_latest = (major_df.sort_values('ПериодОбучения', ascending=False)
                        .groupby('Почта')
                        .head(10)
                        .copy())

    major_stats = major_latest.groupby('Почта').agg({
        'Оценка': [
            ('major_avg_grade', 'mean'),
            ('major_grades_above_4', lambda x: (x >= 4).sum()),
            ('major_grades_below_4', lambda x: (x < 4).sum()),
        ]
    })

    major_last_fail = major_latest.groupby('Почта').apply(get_last_fail_position)
    major_stats[('Оценка', 'major_last_fail_position')] = major_last_fail

    # Combine both statistics
    dc_stats.columns = dc_stats.columns.get_level_values(1)
    major_stats.columns = major_stats.columns.get_level_values(1)

    # Merge the statistics
    combined_stats = pd.merge(
        dc_stats, 
        major_stats, 
        left_index=True, 
        right_index=True, 
        how='outer'
    ).fillna(0)

    # Define the independent exam types
    independent_exam_types = [
        'Независимый экзамен по цифровой грамотности',
        'Независимый экзамен по программированию. Базовый уровень',
        'Независимый экзамен по программированию. Начальный уровень',
        'Независимый экзамен по анализу данных. Базовый уровень',
        'Независимый экзамен по анализу данных. Начальный уровень',
        'Независимый экзамен по программированию. Продвинутый уровень',
        'Независимый экзамен по анализу данных. Продвинутый уровень',
        'Независимый экзамен по анализу данных, искусственному интеллекту и генеративным моделям. Базовый уровень',
        'Независимый экзамен по анализу данных, искусственному интеллекту и генеративным моделям. Начальный уровень',
        'Независимый экзамен по анализу данных, искусственному интеллекту и генеративным моделям. Продвинутый уровень'
    ]

    # Process each independent exam type
    all_stats = []
    for exam_type in independent_exam_types:
        # Create a short name for column prefixes
        short_name = exam_type.replace('Независимый экзамен по ', '').replace(' уровень', '').replace('. ', '_')
        short_name = short_name.replace(' ', '_').replace(',', '').lower()
        
        # Filter data for current exam type
        exam_df = cur_df[cur_df["Дисциплина"] == exam_type].copy()
        exam_latest = (exam_df.sort_values('ПериодОбучения', ascending=False)
                        .groupby('Почта')
                        .head(10)
                        .copy())
        
        # Calculate statistics
        exam_stats = exam_latest.groupby('Почта').agg({
            'Оценка': [
                (f'{short_name}_avg_grade', 'mean'),
                (f'{short_name}_grades_above_4', lambda x: (x >= 4).sum()),
                (f'{short_name}_grades_below_4', lambda x: (x < 4).sum()),
            ]
        })
        
        # Calculate last fail position
        exam_last_fail = exam_latest.groupby('Почта').apply(get_last_fail_position)
        exam_stats[(f'Оценка', f'{short_name}_last_fail_position')] = exam_last_fail
        
        # Clean up column names
        exam_stats.columns = exam_stats.columns.get_level_values(1)
        
        all_stats.append(exam_stats)

    # Merge all statistics
    combined_stats = pd.merge(
        combined_stats, 
        pd.concat([df for df in all_stats], axis=1),
        left_index=True,
        right_index=True,
        how='outer'
    ).fillna(0)

    combined_stats = combined_stats.reset_index()

    dfs[1] = combined_stats

    independent_exams_df = dfs[6][dfs[6]["Дисциплина"].apply(lambda x: "независимый экзамен" in str(x).lower())]

    # Create pivot table with average grades
    result_df = independent_exams_df.pivot_table(
        index='Корп почта',
        columns='Дисциплина',
        values='Оценка',
        aggfunc='mean'
    )

    # Optionally, fill NaN values with 0 or another placeholder
    result_df = result_df.fillna(0)
    result_df = result_df.reset_index()
    # result_df.drop(columns=["Дисциплина"], inplace=True)
    # dfs[6] = result_df

    dfs[6] = result_df

    independent_exams_df = dfs[7][dfs[7]["Дисциплина"].apply(lambda x: "независимый экзамен" in str(x).lower())]

    # Create pivot table with average grades
    result_df = independent_exams_df.pivot_table(
        index='Корп почта',
        columns='Дисциплина',
        values='Оценка',
        aggfunc='mean'
    )

    # Optionally, fill NaN values with 0 or another placeholder
    result_df = result_df.fillna(0)
    result_df = result_df.reset_index()

    dfs[7] = result_df

    email_cols = set(["Адрес электронной почты", "Email address", "Адрес студенческой почты", "Логин", "Почта", "Корп почта"])

    # base_df

    for i in range(len(dfs)):
        for email_col in email_cols:
            if email_col in dfs[i].columns.values:
                # print(len(base_df))
                df_renamed = dfs[i].copy()
                df_renamed.rename(columns={col: f"{col}_{dfs_names[i]}" for col in dfs[i].columns.values}, inplace=True)

                base_df = pd.merge(base_df, df_renamed,
                                    left_on='email',
                                    right_on=email_col + "_" + dfs_names[i],
                                    how='left',
                                    suffixes=('_df1', '_df2'))
                
                base_df.drop(columns=[email_col + "_" + dfs_names[i]], inplace=True)
                # if len(base_df['email'].unique().tolist()) != len(base_df['email'].tolist()):
                #     print(dfs_names[i],dfs[i][email_col].nunique(), len(dfs[i]), "Duplicate emails!!!")
                break

    base_df.head()

    col_patterns = ['test', 'тест', 'оценка', 'variant', 'вариант', 'practice', 'hypothesis', 'практика'
             'гипотеза', 'задание', 'task', 'самопроверка', 'лекция', 'инструмент', 'python', 
             'глава', 'quiz', 'норм', 'сумма', 'перцентиль', 'балл', 'место', 'course', 'курс',
             'кредит', 'экзамен']

    used_cols = [col for col in base_df.columns.values if any([pat in col.split('_', 1)[0].lower() for pat in col_patterns])]
    df_used = base_df[used_cols + ['email']]
    columns_to_drop = [
        'Курс_1',
        'Курс_2',
        'Курс_3',
        'Онлайн-курс_1',
        'Онлайн-курс_2',
        'Онлайн-курс_3',
        'Курс в последнем статусе_Выгрузка по ЕГЭ с 2020 года.xlsx',
        'Курс в последнем статусе_Результаты ЕГЭ.xlsx'
    ]

    df_used.drop(columns=columns_to_drop, inplace=True)

    

    def process_value(x):
        if type(x) == float and math.isnan(x):
            return 0

        if isinstance(x, float):
            return x
        if isinstance(x, str) and x == '':
            return 0
        if isinstance(x, str) and '/' in x:
            a, b = x.split('/')
            a, b = a.strip(), b.strip()
            if a.lower() == 'null' or b.lower() == 'null':
                return 1.0
            return float(a) / float(b)
        
        if isinstance(x, str):
            if x.lower() == 'выполнено':
                return 1.0
            elif x.lower() in ['не выполнено', '-']:
                return 0.0
            x = x.replace(',', '.')
        
        return float(x)

    processed_df = df_used.drop(columns=['email'], inplace=False).apply(lambda x: x.apply(process_value))
    processed_df['email'] = df_used['email']

    # Get all grade columns
    grade_cols = [f'ОЦЕНКА_{i}' for i in range(1, 4)]

    # Create grade_last column by checking each grade column from right to left
    processed_df['target'] = processed_df[grade_cols].replace(0, np.nan).ffill(axis=1).iloc[:, -1]
    processed_df = processed_df[processed_df['target'].notna()]
    
    final_df = pd.DataFrame()
    final_df["email"] = processed_df["email"]
    final_df["target"] = processed_df["target"]

    # Оценки за НЭ

    def get_second_last_non_null(row):
        non_null_values = row.dropna().values
        if len(non_null_values) >= 2:
            return non_null_values[-2]
        return 0.0

    def get_count_passed(row):
        non_null_values = row.dropna().values
        return len([x for x in non_null_values if x >= 4.0])

    def get_count_failed(row):
        non_null_values = row.dropna().values
        return len([x for x in non_null_values if x < 4.0])


    final_df['count_passed'] = processed_df[grade_cols].replace(0.0, np.nan).apply(get_count_passed, axis=1)
    final_df['count_failed'] = processed_df[grade_cols].replace(0.0, np.nan).apply(get_count_failed, axis=1)

    final_df['last_it_grade'] = processed_df[grade_cols].replace(0.0, np.nan).apply(get_second_last_non_null, axis=1)

    grades_sum = processed_df[grade_cols].sum(axis=1)
    grades_count = processed_df[grade_cols].replace(0.0, np.nan).apply(lambda row: row.count(), axis=1)
    final_df['mean_ie_grade'] = grades_sum / grades_count

    # Оценки в целом 
    grades_overall_columns = [
        "Средняя оценка_Рейтинг 21-22 года.csv",
        "Средняя оценка_Рейтинг 22-23 года.csv",
        "Средняя оценка_Рейтинг 23-24 года.csv",
    ]

    def get_grades_mean(row):
        non_zero_values = row[row != 0]
        return non_zero_values.mean() if not non_zero_values.empty else 0.0

    def last_non_zero_value(row):
        non_zero_values = row[row != 0]
        return non_zero_values.iloc[-1] if not non_zero_values.empty else 0.0

    def variance_non_zero(row):
        non_zero_values = row[row != 0]
        var = non_zero_values.var() if not non_zero_values.empty else 0.0
        return var if var != np.nan else 0.0

    def get_grades_min(row):
        non_zero_values = row[row != 0]
        return non_zero_values.min() if not non_zero_values.empty else 0.0


    final_df['grader_overall_mean'] = processed_df[grades_overall_columns].apply(last_non_zero_value, axis=1)
    final_df['last_non_zero'] = processed_df[grades_overall_columns].apply(last_non_zero_value, axis=1)
    final_df['variance_non_zero'] = processed_df[grades_overall_columns].apply(variance_non_zero, axis=1).fillna(0.0)
    final_df['min_non_zero'] = processed_df[grades_overall_columns].apply(get_grades_min, axis=1)

    # Рейтинги
    ratings_overall_columns = [
        "КР сумма_Рейтинг 22-23 года.csv",
        "КР сумма норм_Рейтинг 22-23 года.csv",
        "КРгр сумма норм_Рейтинг 22-23 года.csv",
        "Норм коэф ГР_Рейтинг 22-23 года.csv",
        "Норм коэф_Рейтинг 22-23 года.csv",
        "Сумма кредитов_Рейтинг 22-23 года.csv",
        "Перцентиль_Рейтинг 22-23 года.csv",
        "Минимальный балл_Рейтинг 22-23 года.csv",
        "Место на ОП_Рейтинг 22-23 года.csv",
        "Место в кампусе_Рейтинг 22-23 года.csv",
        "Место на Курсе-ОП_Рейтинг 22-23 года.csv",
        "Место на ОПгр_Рейтинг 22-23 года.csv",
        "КР сумма_Рейтинг 21-22 года.csv",
        "КР сумма норм_Рейтинг 21-22 года.csv",
        "КРгр сумма норм_Рейтинг 21-22 года.csv",
        "Норм коэф ГР_Рейтинг 21-22 года.csv",
        "Норм коэф_Рейтинг 21-22 года.csv",
        "Сумма кредитов_Рейтинг 21-22 года.csv",
        "Перцентиль_Рейтинг 21-22 года.csv",
        "Минимальный балл_Рейтинг 21-22 года.csv",
        "Место на ОП_Рейтинг 21-22 года.csv",
        "Место в кампусе_Рейтинг 21-22 года.csv",
        "Место на Курсе-ОП_Рейтинг 21-22 года.csv",
        "Место на ОПгр_Рейтинг 21-22 года.csv",
        "Место на ОП_Рейтинг 23-24 года.csv",
        "Место в кампусе_Рейтинг 23-24 года.csv",
        "Место на Курсе-ОП_Рейтинг 23-24 года.csv",
        "Место на ОПгр_Рейтинг 23-24 года.csv",
        "Перцентиль_Рейтинг 23-24 года.csv",
        "КР сумма_Рейтинг 23-24 года.csv",
        "КР сумма норм_Рейтинг 23-24 года.csv",
        "КРгр сумма норм_Рейтинг 23-24 года.csv",
        "Норм коэф ГР_Рейтинг 23-24 года.csv",
        "Норм коэф_Рейтинг 23-24 года.csv",
        "Сумма кредитов_Рейтинг 23-24 года.csv",
        "Минимальный балл_Рейтинг 23-24 года.csv",
    ]

    final_df[ratings_overall_columns] = processed_df[ratings_overall_columns]

    # ЕГЭ
    final_df["Балл_ЕГЭ_математика"] = processed_df['Балл ЕГЭ Математика_Результаты ЕГЭ.xlsx']
    final_df["Балл_ЕГЭ_информатика"] = processed_df['Балл ЕГЭ Информатика_Результаты ЕГЭ.xlsx']

    final_df['Балл_ЕГЭ_математика'].hist(bins=30), final_df['Балл_ЕГЭ_информатика'].hist(bins=30)

    final_df["Балл_ЕГЭ_математика"] = final_df["Балл_ЕГЭ_математика"].apply(lambda x: x if x > 20 else 0.0)
    final_df["Балл_ЕГЭ_информатика"] = final_df["Балл_ЕГЭ_информатика"].apply(lambda x: x if x > 20 else 0.0)

    # пройденный материал
 
    not_tests_columns = grade_cols + grades_overall_columns + ratings_overall_columns + ["Балл ЕГЭ_Выгрузка по ЕГЭ с 2020 года.xlsx", "target", "email"]

    only_tests_columns = processed_df.drop(not_tests_columns, axis=1)

    table = []
    for col in only_tests_columns:
        mi = processed_df[col].min()
        ma = processed_df[col].max()
        cnt = processed_df[col].apply(lambda x: x != np.nan).sum()
        su = processed_df[col].sum()
        cnt_zero = processed_df[col].apply(lambda x: x == 0).sum()
        cnt_null = processed_df[col].apply(lambda x: x == np.nan).sum()
        table.append([col, mi, ma, cnt, su, su / cnt, processed_df[col].var(), cnt_zero, cnt_zero / len(processed_df), cnt_null])

    df = pd.DataFrame(table, columns=["name", "min", "max", "cnt", "sum", "avg", "std", "cnt_zero", "cnt_zero_perc", "cnt_null"])

    

    prep_cols = df[df["cnt_zero_perc"] < 0.99]["name"].values
    nums_df = processed_df[prep_cols]

    scaler = StandardScaler()
    nums_df[nums_df.columns] = scaler.fit_transform(nums_df)

    comps = 30

    # Выполняем SVD-разложение
    svd = TruncatedSVD(n_components=comps)  # Выбираем comps компоненты для примера
    svd_coordinates = svd.fit_transform(nums_df)  # Получаем координаты SVD
    svd_columns = [f'SVD_{i+1}' for i in range(svd_coordinates.shape[1])]

    # Создаем DataFrame из координат SVD с теми же индексами, что и исходный DataFrame
    svd_df = pd.DataFrame(svd_coordinates, columns=svd_columns, index=nums_df.index)

    # Объединяем исходный DataFrame с координатами SVD
    nums_svd_df = pd.concat([nums_df, svd_df], axis=1)

    final_df[svd_columns] = nums_svd_df[svd_columns]

    X = final_df.drop(columns=['email', 'target'])
    y = final_df['target']

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_binary = (y_train >= 4).astype(int)
    y_test_binary = (y_test >= 4).astype(int)

    # Initialize model
    gb_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )

    # Train model
    gb_clf.fit(X_train_scaled, y_train_binary)

    # Get staged predictions and calculate metrics for each stage
    train_losses = []
    test_losses = []
    train_aucs = []
    test_aucs = []

    for y_train_pred in gb_clf.staged_predict_proba(X_train_scaled):
        train_losses.append(mean_squared_error(y_train_binary, y_train_pred[:, 1]))
        train_aucs.append(roc_auc_score(y_train_binary, y_train_pred[:, 1]))
        
    for y_test_pred in gb_clf.staged_predict_proba(X_test_scaled):
        test_losses.append(mean_squared_error(y_test_binary, y_test_pred[:, 1]))
        test_aucs.append(roc_auc_score(y_test_binary, y_test_pred[:, 1]))

    # Make final predictions (probabilities)
    y_pred_train_proba = gb_clf.predict_proba(X_train_scaled)[:, 1]
    y_pred_test_proba = gb_clf.predict_proba(X_test_scaled)[:, 1]

    optimal_threshold = np.quantile(y_pred_test_proba, 1 - y_train.sum() / len(y_train))
    # 1 - y_train.sum() / len(y_train) = proportion of negative examples in set

    # raw values
    y_pred_train = gb_clf.predict(X_train_scaled)
    y_pred_test = gb_clf.predict(X_test_scaled)

    # Calculate precision-recall curves
    train_precision, train_recall, train_thresholds = precision_recall_curve(y_train_binary, y_pred_train_proba)
    test_precision, test_recall, test_thresholds = precision_recall_curve(y_test_binary, y_pred_test_proba)

    # Create figure with multiple subplots
    plt.figure(figsize=(20, 12))

    # Plot 1: Learning curves (MSE and AUC)
    plt.subplot(2, 2, 1)
    plt.plot(np.sqrt(train_losses), label='Training MSE', color='blue', linestyle='--')
    plt.plot(np.sqrt(test_losses), label='Test MSE', color='red', linestyle='--')
    plt.plot(train_aucs, label='Training AUC', color='blue')
    plt.plot(test_aucs, label='Test AUC', color='red')
    plt.xlabel('Number of Trees')
    plt.ylabel('Score')
    plt.title('Learning Curves (MSE and AUC)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Confusion Matrix
    plt.subplot(2, 2, 2)
    y_pred_test_binary = (y_pred_test_proba >= optimal_threshold).astype(int)  # using 0.5 as default threshold
    cm = confusion_matrix(y_test_binary, y_pred_test_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Plot 3: ROC curve
    plt.subplot(2, 2, 3)
    # Calculate ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train_binary, y_pred_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_binary, y_pred_test_proba)

    plt.plot(fpr_train, tpr_train, 'b-', 
            label=f'Training (AUC = {roc_auc_score(y_train_binary, y_pred_train_proba):.3f})')
    plt.plot(fpr_test, tpr_test, 'r-', 
            label=f'Test (AUC = {roc_auc_score(y_test_binary, y_pred_test_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot 4: Precision-Recall curve
    plt.subplot(2, 2, 4)
    plt.plot(train_recall, train_precision, 'b-', label='Training')
    plt.plot(test_recall, test_precision, 'r-', label='Test')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate metrics with optimal threshold
    y_pred_train_binary = (y_pred_train_proba >= optimal_threshold).astype(int)
    y_pred_test_binary = (y_pred_test_proba >= optimal_threshold).astype(int)

    print(f"\nOptimal threshold (based on Test F1 score): {optimal_threshold:.4f}")

    print("\nTraining metrics:")
    print(f"Precision: {precision_score(y_train_binary, y_pred_train_binary):.4f}")
    print(f"Recall: {recall_score(y_train_binary, y_pred_train_binary):.4f}")
    print(f"AUC: {roc_auc_score(y_train_binary, y_pred_train_proba):.4f}")

    print("\nTest metrics:")
    print(f"Precision: {precision_score(y_test_binary, y_pred_test_binary):.4f}")
    print(f"Recall: {recall_score(y_test_binary, y_pred_test_binary):.4f}")
    print(f"AUC: {roc_auc_score(y_test_binary, y_pred_test_proba):.4f}")

    # Print regression metrics
    print(f"\nRegression Metrics:")
    print(f"Final Training RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"Final Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

    # Print feature importances
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': gb_clf.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    print("\nTop 10 Most Important Features:")
    print(feature_importance)

    pkl.dump(gb_clf, open('model.pkl', 'wb'))

    return gb_clf

def apply_model(model, data):
    # model = pkl.load(open('model.pkl', 'rb'))
    return model.predict(data)

def check_model(predictions, y_true):
    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    print('Classification Report:')
    print(classification_report(y_true, predictions, digits=4))

    return cm

def get_red_zone(model, data):
    predictions_proba = model.predict_proba(data)[:,1]
    predictions = model.predict(data)
    threshold = np.quantile(predictions_proba, 1 - predictions.sum() / len(predictions))
    predictions_red_zone = predictions_proba < threshold
    predictions_red_zone.sort_values(inplace=True, ascending=True)
    return predictions_red_zone

