import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from time import time
import numpy as np
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import scipy.stats as st  
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import reciprocal
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

def age_category(age):
    if age <= 12:
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

def label_encoding(df, feature):
    labelEncoder = LabelEncoder()
    df[feature] = labelEncoder.fit_transform(df[feature])
    return df

def model_svm(df_dirty, df_original):
    continuous_features = ['age', 'bmi', 'avg_glucose_level']
    binary_features = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'smoking_status']
    categorical_features = ['work_type']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    binary_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ]
    )

    X_original = df_original.drop('stroke', axis=1)
    y_original = df_original['stroke']
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

    X_dirty = df_dirty.drop('stroke', axis=1)
    y_dirty = df_dirty['stroke']
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=42)

    svm_model = SVC(kernel='rbf', probability=True, random_state=0)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', svm_model)
    ])

    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': [1, 0.1, 0.01, 0.001]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)
    grid_search.fit(X_train_dirty, y_train_dirty)

    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    y_pred_original = grid_search.predict(X_test_original)
    print("Classification Report on Original Test Set:")
    print(classification_report(y_test_original, y_pred_original))

    plot_roc_curve_svm(y_test_original, grid_search, X_test_original)
    plt.show()

    plot_confusion_matrix(y_test_original, y_pred_original)
    plt.show()

    return grid_search

def plot_roc_curve_svm(y_test, classifier, X_test):
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]  
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    print("AUC Score:", roc_auc)

def plot_confusion_matrix(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    labels = [1, 0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
