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
        duplicate_count = df[column].duplicated().sum()
        duplicate_percentage = (duplicate_count / total_rows) * 100
        
        print(f"\nFeature: {column}")
        print(f"Null Count: {null_count}")
        print(f"Null Percentage: {null_percentage:.2f}%")
        print(f"Duplicate Count: {duplicate_count}")
        print(f"Duplicate Percentage: {duplicate_percentage:.2f}%")



    
def check_categorical_values (df):
    categorical_features = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for feature in categorical_features:
        abnormal_values = (df[feature] != 0) & (df[feature] != 1)
        if abnormal_values.any():
            print('abnormal values present')
            print(df[abnormal_values])
        else:
            print('correct values')
            
