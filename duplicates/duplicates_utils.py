import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def drop_negative_age(df):
    df[df['age'] >= 0]

def drop_null_values(df):
    df.dropna(inplace=True)
    #df.dropna()
    return df

def drop_negative_values(df, feature):
    abnormal_values = (df[feature] < 0)
    df_c = df.drop(df[abnormal_values].index)
    return df_c

def print_duplicates_values(df):
    total_rows = len(df)
    print("Number of rows: ", total_rows)
    duplicate_counts = df.duplicated().sum()
    print("Duplicate Counts: ", duplicate_counts)
    duplicate_percentage = (duplicate_counts / total_rows) * 100
    print("Percentage of Duplicate Values: ", duplicate_percentage)

def duplicates_values(df, percentage):
    num_duplicates = int(len(df) * (percentage / 100))
    indices_to_duplicate = np.random.choice(df.index, size=num_duplicates, replace=True)
    duplicated_data = df.loc[indices_to_duplicate]
    df_with_duplicates = pd.concat([df, duplicated_data], ignore_index=True)
    return df_with_duplicates