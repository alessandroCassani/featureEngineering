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
    null_counts = df.isnull().sum()
    print("Null Value Counts:")
    print(null_counts)

    # Count duplicates
    duplicate_counts = df.duplicated().sum()
    print("\nDuplicate Counts:")
    print(duplicate_counts)
    
    
def check_categorical_values (df):
    categorical_features = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for feature in categorical_features:
        abnormal_values = (df[feature] != 0) & (df[feature] != 1)
        if abnormal_values.any():
            print('abnormal values present')
            print(df[abnormal_values])
        else:
            print('correct values')