import pandas as pd
from sklearn.preprocessing import LabelEncoder

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obesity'

def categorize_bmi(df):
    df['bmi'] = df['bmi'].apply(bmi_category)
    
    return df

def glucose_category(glucose_level):

    if glucose_level < 70:
        return 'Low'
    elif 70 <= glucose_level < 100:
        return 'Normal'
    elif 100 <= glucose_level < 125:
        return 'Prediabetes'
    else:
        return 'Diabetes'

def categorize_glucose(df):
    df['avg_glucose_level'] = df['avg_glucose_level'].apply(glucose_category)
    
    return df

def label_encoding(df,feature):
    labelEncoder = LabelEncoder()
    df[feature] = labelEncoder.fit_transform(df[feature])
    return df

def age_category(age):
    if age < 0:
        return "Invalid age"
    elif age <= 12:
        return "Child"
    elif age <= 19:
        return "Teenager"
    elif age <= 64:
        return "Adult"
    else:
        return "Senior"

def categorize_age(df):
    df['age'] = df['age'].apply(age_category)
    return df
