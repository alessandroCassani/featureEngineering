import random
import pandas as pd

def modify_values(column, dataset, percentage):
    num_rows = int(len(dataset) * (percentage / 100))
    rows_to_modify = random.sample(range(len(dataset)), num_rows)
    rows_modified = 0
    original_dataset = dataset.copy()
    if column == "work_type": 
        for i, value in enumerate(column):
            if i not in rows_to_modify:
                continue
            if pd.notnull(value):
                if value == 0:
                    modified_value = value + 1  # Increment for 0
                elif value == 4:
                    modified_value = value - 1  # Decrement for 4
                elif 1 <= value <= 3:
                    modified_value = value + random.choice([-1, 1])  # Increment or decrement for 1, 2, 3
                dataset.loc[i, column.name] = modified_value
                rows_modified += 1
    elif pd.api.types.is_integer_dtype(column):
        for i, value in enumerate(column):
            if i not in rows_to_modify:
                continue
            if pd.notnull(value) and value != 0 and value != 1:  # Skip binary features
                modified_value = int(value * 1.2)  # Increase the value by 20%
                dataset.loc[i, column.name] = modified_value
                rows_modified += 1

    elif pd.api.types.is_float_dtype(column):
        for i, value in enumerate(column):
            if i not in rows_to_modify:
                continue
            if pd.notnull(value) and (value != 0.0 and value != 1.0):  # Skip binary features
                modified_value = value * 1.2  # Increase the value by 20%
                dataset.loc[i, column.name] = modified_value
                rows_modified += 1

    return dataset, original_dataset

