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

# maybe it makes delete - uniform
def add_outlier_continuous(df, feature, percentage):
    mean = df[feature].mean()
    std = df[feature].std()

    z_score = 3
    lower_limit = mean - (z_score * std)
    upper_limit = mean + (z_score * std)

    n_rows = int(len(df) * (percentage / 100))

    outlier_indices = np.random.choice(df.index, size=n_rows, replace=False)

    outliers = np.random.uniform(lower_limit, upper_limit, size=n_rows)
    outliers = np.abs(outliers)  # Module to avoid null values
    outliers = np.round(outliers).astype(int)  # Approximates values to integers

    df.loc[outlier_indices, feature] = outliers
    return df

# variable normal
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
    
def drop_negative_values(df, feature):
    abnormal_values = (df[feature] < 0)
    df_c = df.drop(df[abnormal_values].index)
    return df_c

def drop_negative_age(df):
    df[df['age'] >= 0]

def add_categorical_outliers(feature, percentage, df):
    # Calcola il valore outlier per la colonna specificata
    outlier_value = df[feature].value_counts().idxmin()
    print("Valore meno frequente: ", outlier_value)

    # Determina il numero di righe da modificare
    n_rows = int(len(df) * (percentage / 100))

    # Seleziona casualmente le righe da modificare
    rows_to_modify = np.random.choice(df.index, size=n_rows, replace=False)

    # Inserisci il valore outlier nella percentuale di righe specificata
    df.loc[rows_to_modify, feature] = outlier_value
    return df

def detect_outliers_categorical(df, feature, threshold):
    # Calcola la frequenza delle categorie
    value_counts = df[feature].value_counts()
    total_count = len(df[feature])
    # Identifica le categorie che sono sotto la soglia (percentuale)
    outliers = value_counts[value_counts / total_count < threshold].index
    print(f"Value counts:\n{value_counts}")  # Debug: stampa delle frequenze
    print(f"Outliers detected: {outliers.tolist()}")  # Debug: stampa degli outlier
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
