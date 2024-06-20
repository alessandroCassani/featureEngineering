import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def print_duplicates_values(df):
    total_rows = len(df)
    print("Number of rows: ", total_rows)
    duplicate_counts = df.duplicated().sum()
    print("Duplicate Counts: ", duplicate_counts)
    duplicate_percentage = (duplicate_counts / total_rows) * 100
    print("Percentage of Duplicate Values: ", duplicate_percentage)

def replace_duplicates_values(df, percentage):
    num_duplicates = int(len(df) * (percentage / 100))
    indices_to_drop = np.random.choice(df.index, size=num_duplicates, replace=False)
    df_removed = df.drop(indices_to_drop).reset_index(drop=True)
    indices_to_duplicates = np.random.choice(df_removed.index, size=num_duplicates, replace=True)
    duplicated_data = df_removed.loc[indices_to_duplicates]
    df_with_duplicates = pd.concat([df_removed, duplicated_data], ignore_index=True)

    return df_with_duplicates
