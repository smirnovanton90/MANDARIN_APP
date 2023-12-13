import pandas as pd
from datetime import datetime as dt

# функция создает dataFrame из одной строки со столбцами, список которых передан в переменной columns


def df_structure(columns):
    keys = columns
    values = [0]*len(columns)
    data = dict(zip(keys, values))
    df = pd.DataFrame(data, index=['client'])
    return df


# функция считает количество лет с указанной даты до текущего момента
def year_counter(date):
    return dt.now().year - dt.strptime(date[:-1], '%Y-%m-%d %H:%M:%S.%f').year


# функция заполняет dataFrame данными для передачи в модель
def data_transform(columns, json):

    request = df_structure(columns)

    request.at['client', 'Age'] = year_counter(json['birth_date'])
    request.at['client', 'Seniority'] = year_counter(json['job_start_date'])
    request.at['client', 'DebtRatio'] = (json['loan_amount'] / json['loan_term']) / \
                                        (json['month_profit'] - json['month_expense'])
    if json['gender'] == 0:
        request.at['client', 'Gender_Male'] = 1
    else:
        request.at['client', 'Gender_Female'] = 1

    if json['snils'] == 0:
        request.at['client', 'SNILS_Empty'] = 1
    else:
        request.at['client', 'SNILS_Not_empty'] = 1

    if json['child_count'] == 0:
        request.at['client', 'ChildCount_0.0'] = 1
    elif json['child_count'] == 1:
        request.at['client', 'ChildCount_1.0'] = 1
    elif json['child_count'] == 2:
        request.at['client', 'ChildCount_2.0'] = 1
    elif json['child_count'] == 3:
        request.at['client', 'ChildCount_3.0'] = 1
    elif json['child_count'] == 4:
        request.at['client', 'ChildCount_4.0'] = 1
    elif json['child_count'] == 5:
        request.at['client', 'ChildCount_5.0'] = 1

    ohe_data = [json['education'], json['employment_status'], json['family_status'], json['merch_code'],
                json['goods_category']]
    for data in ohe_data:
        for column in request.columns:
            if data in column:
                request.at['client', column] = 1

    return request


def predictor(model, df):
    prediction = round(model.predict_proba(df)[0][1] * 100, 2)
    return prediction
