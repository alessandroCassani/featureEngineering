import pandas as pd
import numpy as np

def insert_high_values_for_stroke(df, percentage):

    high_values = {
    'bmi': 35,  # High BMI value
    'avg_glucose_level': 250  # High average glucose level value
    }
    num_entries_to_modify = int(len(df) * (percentage / 100))
    
    # Randomly select indices to modify
    indices_to_modify = np.random.choice(df.index, size=num_entries_to_modify, replace=False)
    
    # Dictionary to store original values
    original_values = {column: df.loc[indices_to_modify, column].copy() for column in high_values.keys()}
    
    # Modify the selected entries
    for column, high_value in high_values.items():
        df.loc[indices_to_modify, column] = high_value
    
    return df, indices_to_modify, original_values

def restore_original_values(df, original_values):
    for column, values in original_values.items():
        indices, original_vals = values.index, values.values
        df.loc[indices, column] = original_vals
    return df


