import pandas as pd
import numpy as np

def introduce_age_married_inconsistencies(df, percentage):
    num_rows_to_modify = int(len(df) * percentage / 100)
    
    rows_to_modify = df.sample(n=num_rows_to_modify, random_state=42).index
    
    original_values = df.loc[rows_to_modify, ['age', 'ever_married']].copy()
    
    df.loc[rows_to_modify, 'age'] = np.random.randint(0, 16, size=num_rows_to_modify)
    df.loc[rows_to_modify, 'ever_married'] = 1
    
    return original_values, rows_to_modify

def introduce_age_workType_inconsistencies(df, percentage):
    num_rows_to_modify = int(len(df) * percentage / 100)
    
    rows_to_modify = df.sample(n=num_rows_to_modify, random_state=42).index

    original_values = df.loc[rows_to_modify, ['age', 'work_type']].copy()
    
    df.loc[rows_to_modify, 'age'] = np.random.randint(0, 18, size=num_rows_to_modify)
    df.loc[rows_to_modify, 'work_type'] = np.random.randint(2, df['work_type'].max() + 1, size=num_rows_to_modify)
    
    return original_values, rows_to_modify


def introduce_negative_ages(df, percentage):
    num_rows_to_modify = int(len(df) * percentage / 100)
    rows_to_modify = df.sample(n=num_rows_to_modify, random_state=42).index
    
    original_values = df.loc[rows_to_modify, 'age'].copy()
    df.loc[rows_to_modify, 'age'] = -1
    return rows_to_modify, original_values
    
    
def check_negative_values (df, feature):
    abnormal_values = (df[feature] < 0)
    if abnormal_values.any():
        print(f'abnormal values present in {feature} feature')
        print(df[abnormal_values])
    else:
        print(f'correct values in {feature} feature')
        

def check_age_married_consistency(df):
    invalid_rows_index = df[(df['age'] < 16) & (df['ever_married'] == 1)].index
    #df = df.drop(invalid_rows_index, axis=0)
    print('number of incosistencies: \n')
    print(len(invalid_rows_index))
    #print("Rows with age < 16 and ever_married == 1 have been dropped")
    
def check_age_workType_consistency(df):
    invalid_rows_index = df[(df['age'] < 16) & ((df['work_type'] != 0) | (df['work_type'] != 1))].index
    #df = df.drop(invalid_rows_index, axis=0)
    print('number of incosistencies: \n')
    print(len(invalid_rows_index))
    #print("Rows with age < 16 and work_type different from 0 or 1 dropped")