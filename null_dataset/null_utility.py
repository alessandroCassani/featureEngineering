import numpy as np

def drop_negative_age(df):
    df[df['age'] >= 0]

def add_null_values(df, column_name, percentage):
    num_nulls = int(len(df) * (percentage / 100))
    indices_to_nullify = np.random.choice(df.index, size=num_nulls, replace=False)
    original_values = df.loc[indices_to_nullify, column_name].copy()
    if df[column_name].dtype == 'int64':
        df.loc[indices_to_nullify, column_name] = -999
    elif df[column_name].dtype == 'float':
        df.loc[indices_to_nullify, column_name] = -999.9
    return indices_to_nullify, original_values