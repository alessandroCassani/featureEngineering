import random
import pandas as pd

def modify_values(column_name, dataset, percentage):
    # Ensure the dataset index is a RangeIndex (i.e., from 0 to len(dataset)-1)
    if not isinstance(dataset.index, pd.RangeIndex):
        dataset = dataset.reset_index(drop=True)
    
    num_rows = int(len(dataset) * (percentage / 100))
    rows_to_modify = random.sample(range(len(dataset)), num_rows)
    original_dataset = dataset.copy()

    print(f"Modifying {num_rows} rows in column '{column_name}'")
    print(f"Rows to modify: {rows_to_modify}")
    
    if column_name == "work_type":
        for i in rows_to_modify:
            value = dataset.loc[i, column_name]
            if pd.notnull(value):
                if value == 0:
                    modified_value = value + 1  # Increment for 0
                elif value == 4:
                    modified_value = value - 1  # Decrement for 4
                elif 1 <= value <= 3:
                    modified_value = value + random.choice([-1, 1])  # Increment or decrement for 1, 2, 3
                dataset.loc[i, column_name] = modified_value

    elif pd.api.types.is_integer_dtype(dataset[column_name]):
        for i in rows_to_modify:
            value = dataset.loc[i, column_name]
            if pd.notnull(value) and value != 0 and value != 1:  # Skip binary features
                modified_value = int(value * 1.3)  # Increase the value by 10%
                dataset.loc[i, column_name] = modified_value

    elif pd.api.types.is_float_dtype(dataset[column_name]):
        for i in rows_to_modify:
            value = dataset.loc[i, column_name]
            if pd.notnull(value) and value != 0.0 and value != 1.0:  # Skip binary features
                modified_value = value * 1.3  # Increase the value by 10%
                dataset.loc[i, column_name] = modified_value

    return dataset  # Return the modified dataset
