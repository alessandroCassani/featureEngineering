import numpy as np

def drop_negative_age(df):
    df[df['age'] >= 0]

def add_null_values(df, column_name, percentage):
    num_nulls = int(len(df) * (percentage / 100))
    indices_to_nullify = np.random.choice(df.index, size=num_nulls, replace=False)
    original_values = df.loc[indices_to_nullify, column_name].copy()
    df.loc[indices_to_nullify, column_name] = np.nan
    return indices_to_nullify, original_values

def print_null_values(df):
    total_rows = len(df)
    print(f'Total rows: {total_rows}')
    
    print("Null Value Counts:")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    print("\nPercentage of Null Values:")
    null_percentage = (null_counts / total_rows) * 100
    print(null_percentage)
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        
        print(f"\nFeature: {column}")
        print(f"Null Count: {null_count}")
        print(f"Null Percentage: {null_percentage:.2f}%")