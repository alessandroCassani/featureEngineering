import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def max_min_commonValue(df):
    for column in df.columns:
        min_value = df[column].min()
        max_value = df[column].max()
        most_common_value = df[column].mode()[0]
    
        print(column.upper())
        print("min value:" + str(min_value))
        print("max value: " + str(max_value))
        print("max common value: " + str(most_common_value) + "\n")

def print_null_duplicates_values(df):
    total_rows = len(df)
    print(f'total rows: {total_rows}')
    
    print("Null Value Counts:")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    print("\nPercentage of Null Values:")
    null_percentage = (null_counts / total_rows) * 100
    print(null_percentage)
    
    print("\nDuplicate Counts:")
    duplicate_counts = df.duplicated().sum()
    print(duplicate_counts)
    
    print("\nPercentage of Duplicate Values:")
    duplicate_percentage = (duplicate_counts / total_rows) * 100
    print(duplicate_percentage)
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        
        print(f"\nFeature: {column}")
        print(f"Null Count: {null_count}")
        print(f"Null Percentage: {null_percentage:.2f}%")

    
def check_categorical_values(df):
    categorical_features = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    flag = True
    for feature in categorical_features:
        if feature != 'work_type':
            abnormal_values = (df[feature] != 0) & (df[feature] != 1)
        else:
            abnormal_values = (df[feature] < 0) | (df[feature] > 4)
        if abnormal_values.any():
            flag = False
            print(f"Abnormal values found in feature '{feature}':")
            print(df[abnormal_values][[feature]])
            print("\n")
           
    if flag == False:
        print('Abnormal values present')
    else:
        print('All values are correct')
            
            
def check_negative_values (df, feature):
    abnormal_values = (df[feature] < 0)
    if abnormal_values.any():
        print(f'abnormal values present in {feature} feature')
        print(df[abnormal_values])
    else:
        print(f'correct values in {feature} feature')
        

def check_age_married_consistency(df):
    invalid_rows_index = df[(df['age'] < 16) & (df['ever_married'] == 1)].index
    #df = df.drop(invalid_rows_index, axis=0)
    print('number of incosistencies: \n')
    print(len(invalid_rows_index))
    #print("Rows with age < 16 and ever_married == 1 have been dropped")
    
def check_age_workType_consistency(df):
    invalid_rows_index = df[(df['age'] < 18) & ((df['work_type'] != 0) | (df['work_type'] != 1))].index
    #df = df.drop(invalid_rows_index, axis=0)
    print('number of incosistencies: \n')
    print(len(invalid_rows_index))
    #print("Rows with age < 16 and work_type different from 0 or 1 dropped")
    
def detect_outliers_zscore(df, threshold):
    z_scores = np.abs((df - df.mean()) / df.std())
    return z_scores > threshold

def visualize_outliers(df):
    threshold=3
    numerical_features = ['age', 'avg_glucose_level','bmi']
    for feature in numerical_features:
        outliers = detect_outliers_zscore(df[feature], threshold)
        if outliers.any():
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

def drop_negative_age(df):
    df[df['age'] >= 0]

def add_null_values(df, column_name, percentage):
    num_nulls = int(len(df) * (percentage / 100))
    indices_to_nullify = np.random.choice(df.index, size=num_nulls, replace=False)
    original_values = df.loc[indices_to_nullify, column_name].copy()
    df.loc[indices_to_nullify, column_name] = np.nan
    return indices_to_nullify, original_values
