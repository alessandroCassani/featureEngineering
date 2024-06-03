import pandas as pd
import numpy as np

def introduce_inconsistencies(df, percentage):
    single_percentage = percentage / 3
    age_married_originals = introduce_age_married_inconsistencies(df, single_percentage)
    age_workType_originals = introduce_age_workType_inconsistencies(df, single_percentage)
    negative_ages_originals = introduce_negative_ages(df, single_percentage)
    
    all_originals = {**age_married_originals, **age_workType_originals, **negative_ages_originals}
    
    return all_originals

def introduce_age_married_inconsistencies(df, percentage):
    num_rows_to_modify = int(len(df) * percentage / 100)
    rows_to_modify = df.sample(n=num_rows_to_modify, random_state=42).index
    original_values = df.loc[rows_to_modify, ['age', 'ever_married']].copy()

    df.loc[rows_to_modify, 'age'] = np.random.randint(0, 16, size=num_rows_to_modify)
    df.loc[rows_to_modify, 'ever_married'] = 1

    return {index: original_values.loc[index].to_dict() for index in rows_to_modify}

def introduce_age_workType_inconsistencies(df, percentage):
    num_rows_to_modify = int(len(df) * percentage / 100)
    rows_to_modify = df.sample(n=num_rows_to_modify, random_state=42).index
    original_values = df.loc[rows_to_modify, ['age', 'work_type']].copy()

    df.loc[rows_to_modify, 'age'] = np.random.randint(0, 18, size=num_rows_to_modify)
    df.loc[rows_to_modify, 'work_type'] = np.random.randint(2, df['work_type'].max() + 1, size=num_rows_to_modify)

    return {index: original_values.loc[index].to_dict() for index in rows_to_modify}

def introduce_negative_ages(df, percentage):
    num_rows_to_modify = int(len(df) * percentage / 100)
    rows_to_modify = df.sample(n=num_rows_to_modify, random_state=42).index
    original_values = df.loc[rows_to_modify, 'age'].copy()

    df.loc[rows_to_modify, 'age'] = -1
    
    return {index: {'age': original_values.loc[index]} for index in rows_to_modify}

def restore_original_values(df, original_values_dict):
    for index, original_values in original_values_dict.items():
        for col, value in original_values.items():
            df.at[index, col] = value

def visualize_inconsistencies(df):
    total_inconsistency_percentage = check_age_married_consistency(df) + check_age_workType_consistency(df) + check_negative_age_values(df)
    print('\nTOTAL INCONSISTENCY PERCENTAGE')
    print(total_inconsistency_percentage)
    
def check_negative_age_values(df):
    abnormal_values = df['age'] < 0
    num_abnormal_values = abnormal_values.sum()
    total_values = len(df)
    percentage_abnormal_values = (num_abnormal_values / total_values) * 100

    if num_abnormal_values > 0:
        print(f'Number of abnormal values: {num_abnormal_values}')
        print(f'Percentage of abnormal values: {percentage_abnormal_values:.2f}%')
    else:
        print('All values in age feature are correct.')
        
    return percentage_abnormal_values
        

def check_age_married_consistency(df):
    invalid_rows_index = df[(df['age'] < 16) & (df['ever_married'] == 1)].index
    num_inconsistencies = len(invalid_rows_index)
    percentage_inconsistencies = (num_inconsistencies / len(df)) * 100
    print(f'Number of inconsistencies in age and married features: {num_inconsistencies}')
    print(f'Percentage of inconsistencies in age and married features: {percentage_inconsistencies:.2f}%')
    return percentage_inconsistencies

def check_age_workType_consistency(df):
    invalid_rows_index = df[(df['age'] < 16) & (df['work_type'] != 0) & (df['work_type'] != 1)].index
    num_inconsistencies = len(invalid_rows_index)
    percentage_inconsistencies = (num_inconsistencies / len(df)) * 100
    print(f'Number of inconsistencies in age and workType features: {num_inconsistencies}')
    print(f'Percentage of inconsistencies in age and workType features: {percentage_inconsistencies:.2f}%')
    return percentage_inconsistencies