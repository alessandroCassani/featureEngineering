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

def insert_high_values_for_stroke_label(df, percentage, stroke_bool):
    high_values = {
        'bmi': 35,  # High BMI value
        'avg_glucose_level': 250  # High average glucose level value
    }
    
    if stroke_bool:
        stroke_indices = df[df['stroke'] == 1].index
    else:
        stroke_indices = df[df['stroke'] == 0].index
        
    num_entries_to_modify = int(len(stroke_indices) * (percentage / 100))
    
    # Randomly select indices to modify among the stroke entries
    indices_to_modify = np.random.choice(stroke_indices, size=num_entries_to_modify, replace=False)
    
    # Dictionary to store original values
    original_values = {column: df.loc[indices_to_modify, column].copy() for column in high_values.keys()}
    
    # Modify the selected entries
    for column, high_value in high_values.items():
        df.loc[indices_to_modify, column] = high_value
    
    return  df,indices_to_modify, original_values


