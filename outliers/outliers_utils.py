import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def detect_outliers_zscore(df, feature, threshold):
    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
    return z_scores > threshold

def visualize_outliers_specific(df, feature):
    threshold = 3
    outliers = detect_outliers_zscore(df, feature, threshold)
    if outliers.any():
        print("Outliers found:")
        plt.figure(figsize=(8, 4))
        sns.histplot(df[feature], kde=True, color='blue', bins=30)
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.axvline(x=df[feature][outliers].min(), color='red', linestyle='--', label='Outliers')
        plt.axvline(x=df[feature][outliers].max(), color='red', linestyle='--')
        plt.legend()
        plt.show()
    else:
        print('no outliers detected')

# Function to replace outliers, using variable normal
def outliers_replace(df, feature, percentage):
    # Removed rows from the dataset
    num_rows = int(len(df) * (percentage / 100))
    rows_drop = np.random.choice(df.index, size=num_rows, replace=False)
    df_removed = df.drop(rows_drop)

    # Computation of outliers
    mean = df_removed[feature].mean()
    std_dev = df_removed[feature].std()

    lower_outliers = np.abs(np.random.normal(mean + 1 * std_dev, std_dev, size=num_rows // 2))
    upper_outliers = np.abs(np.random.normal(mean + 2 * std_dev, std_dev, size=num_rows // 2))
    outliers_values = np.concatenate([lower_outliers, upper_outliers])

    # If the number of outliers is not even, it add an extra outlier to reach the correct number
    if num_rows % 2 != 0:
        extra_outlier = np.random.uniform(mean + 2 * std_dev, std_dev, size=1)
        outliers_values = np.append(outliers_values, extra_outlier)

    outliers = df.sample(n=len(outliers_values)).reset_index(drop=True)
    outliers[feature] = outliers_values
    df_outliers_added = pd.concat([df_removed, outliers], ignore_index=True)

    return df_outliers_added
    
def add_categorical_outliers(feature, percentage, df):
    # Calculates the outlier value for the specified column
    outlier_value = df[feature].value_counts().idxmin()
    print("Valore meno frequente: ", outlier_value)

    n_rows = int(len(df) * (percentage / 100))

    rows_to_modify = np.random.choice(df.index, size=n_rows, replace=False)

    # Enter the outlier value in the specified percentage of rows
    df.loc[rows_to_modify, feature] = outlier_value
    return df

def detect_outliers_categorical(df, feature, threshold):
    value_counts = df[feature].value_counts()
    total_count = len(df[feature])
    outliers = value_counts[value_counts / total_count < threshold].index
    print(f"Value counts:\n{value_counts}")  
    print(f"Outliers detected: {outliers.tolist()}")
    return outliers

def visualize_outliers_categorical(df, feature):
    threshold = 3
    outliers = detect_outliers_categorical(df, feature, threshold)
    
    if len(outliers) > 0:
        print("Outliers found:")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
        plt.title(f'Count Plot of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        
        plt.xticks(rotation=45)
        plt.show()
    else:
        print('No outliers detected')

# Function to replace outliers with values 20% above the mean calculated from the original dataframe
def replace_outliers_with_above_mean_original(df_original, df_dirty, feature):
    threshold = 3
    outliers_idx = detect_outliers_zscore(df_dirty, feature, threshold)
    if len(outliers_idx) == 0:
        print("No outliers detected.")
        return df_dirty

    mean_value = df_original[feature].mean()  
    replacement_value = mean_value * 1.20

    df_dirty.loc[outliers_idx, feature] = replacement_value
    return df_dirty