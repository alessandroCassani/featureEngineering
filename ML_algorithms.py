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


def plot_decision_tree(tree_model, feature_names, class_names=['0', '1']):
    plt.figure(figsize=(20, 10))
    tree_plot = plot_tree(tree_model, filled=True, feature_names=feature_names, class_names=class_names, rounded=True, impurity=False, fontsize=8)
    # Get the labels of each node and display them
    text = tree_plot[0]
    print("Node Labels:\n", text)
    plt.show()

def plot_roc_curve(y_test, classifier, X_test):
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
    return y_pred_prob, y_test
    
    
def plot_feature_importance_decision_tree(best_tree_classifier, X):
    importance = best_tree_classifier.feature_importances_
    # Sort feature importance
    sorted_idx = np.argsort(importance)
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[sorted_idx], align='center')
    plt.yticks(range(len(importance)), [X.columns[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.show()
        
def plot_confusion_matrix(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    labels = [1, 0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

    
def k_fold_cross_validation_dt(model, df):
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    n_fold=10
    
    folds = KFold(n_splits=n_fold, shuffle=True)

    accuracy_k_fold_dt = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_test, y_test)):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        # Make predictions on the validation set and calculate accuracy
        y_valid_pred = model.predict(X_valid_fold)
        accuracy_k_fold_dt.append(accuracy_score(y_valid_fold, y_valid_pred))

    print("Accuracy for each fold:", accuracy_k_fold_dt)
    print("Mean accuracy:", np.mean(accuracy_k_fold_dt))

    # Calculate the 95% confidence interval
    confidence_interval = st.t.interval(0.95, df=len(accuracy_k_fold_dt)-1, loc=np.mean(accuracy_k_fold_dt), scale=st.sem(accuracy_k_fold_dt))
    print("95% confidence interval:", confidence_interval)
    
    # Plot the accuracy for each fold with the mean and confidence interval
    mean_accuracy = np.mean(accuracy_k_fold_dt)  # Calculate mean accuracy
    # Plot the mean and confidence interval
    plt.errorbar(1, mean_accuracy, yerr=(confidence_interval[1] - confidence_interval[0])/2, fmt='o')
    # Add labels and title
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.title('Mean with Confidence Interval')
    # Show the plot
    plt.legend()
    plt.show()

def model_svm(df_dirty, df_original):
    continuous_features = ['age', 'bmi', 'avg_glucose_level']
    binary_features = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'smoking_status']  
    categorical_features = ['work_type']
    
  
    continuous_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
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
    class_report = classification_report(y_test_original, y_pred_original)
    print(class_report)
    y_pred_prob, y_test = plot_roc_curve_svm(y_test_original, grid_search, X_test_original)
    plt.show()
    plot_confusion_matrix(y_test_original, y_pred_original)
    plt.show()
    return y_pred_prob, y_test, class_report, grid_search
    
def model_dt(df_dirty, df_clean):
    # Splitting the dataset into features and target variable
    X_dirty = df_dirty.drop('stroke', axis=1)
    y_dirty = df_dirty['stroke']
    
    X_original = df_clean.drop('stroke', axis=1)
    y_original = df_clean['stroke']

    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

    # Splitting the dirty dataset into training set and test set (with 30% testing)
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=42)
    
    decision_tree_model = DecisionTreeClassifier(max_depth=10, random_state=0)
    decision_tree_model.fit(X_train_dirty, y_train_dirty)
    y_pred_original = decision_tree_model.predict(X_test_original)
    
    # Printing performance on the test set original
    print("Classification Report on Test Set - original:")
    class_report = classification_report(y_test_original, y_pred_original)
    print(class_report)
    
    plot_decision_tree(decision_tree_model,df_clean.columns,)
    plot_feature_importance_decision_tree(decision_tree_model, X_train_dirty)
    y_pred_prob, y_test = plot_roc_curve(y_test_original, decision_tree_model, X_test_original)
    plot_confusion_matrix(y_test_original, y_pred_original)
    return y_pred_prob, y_test, class_report, decision_tree_model


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
    return y_pred_prob, y_test

def plot_roc_curve_conlusion(y_pred_prod_dt, y_test_dt, y_pred_prod_svm, y_test_svm):
    fpr1, tpr1, thresholds1 = roc_curve(y_test_dt, y_pred_prod_dt)
    roc_auc1 = roc_auc_score(y_test_dt, y_pred_prod_dt)
    fpr2, tpr2, thresholds2 = roc_curve(y_test_svm, y_pred_prod_svm)
    roc_auc2 = roc_auc_score(y_test_svm, y_pred_prod_svm)
    plt.figure(figsize=(10, 8))

    # Model 1 - Decision Tree
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='Decision Tree(AUC = %0.2f)' % roc_auc1)

    # Model 2 - Support Vector Machine
    plt.plot(fpr2, tpr2, color='blue', lw=2, label='Support Vector Machine(AUC = %0.2f)' % roc_auc2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Graphic
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()


def plot_confidence_intervals(model_results):
    plt.figure(figsize=(10, 8))
    
    for i, (model_name, mean_accuracy, confidence_interval) in enumerate(model_results):
        plt.errorbar(i, mean_accuracy, yerr=(confidence_interval[1] - confidence_interval[0]) / 2, fmt='o', label=model_name)
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.title('Mean with Confidence Interval')
    plt.xticks(range(len(model_results)), [name for name, _, _ in model_results])
    plt.legend()
    plt.show()

def plot_roc_curve_conclusion_with_results(roc_results):
    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'blue', 'red', 'green', 'yellow', 'purple', 'pink']
    
    for i, (y_pred_prob, y_test, feature) in enumerate(roc_results):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label='{0} (AUC = {1:.2f})'.format(feature, roc_auc))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Graphic
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    plt.show()

def decision_tree(df):
    # Splitting the dataset con duplicates into features and target variable
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Splitting the dataset into training set and test set (with 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    decision_tree_model = DecisionTreeClassifier(max_depth=10, random_state=0)
    decision_tree_model.fit(X_train, y_train)
    print("\n--- Prestazioni del modello Decision Tree applicato al set di Test: \n")
    pred_test = decision_tree_model.predict(X_test)

    # Printing performance on the test set
    print("Classification Report on Test Set:")
    class_report_test = classification_report(y_test, pred_test)
    print(class_report_test)
    
    plot_decision_tree(decision_tree_model,df.columns)
    plot_feature_importance_decision_tree(decision_tree_model, X_train)
    y_pred_prob, y_test = plot_roc_curve(y_test, decision_tree_model, X_test)
    plot_confusion_matrix(y_test, pred_test)
    return y_pred_prob, y_test, class_report_test, decision_tree_model

def SVM(df):
    continuous_features = ['age', 'bmi', 'avg_glucose_level']
    binary_features = ['sex', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'smoking_status']  
    categorical_features = ['work_type']
    
  
    continuous_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ]
    )

    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
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
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")
    y_pred_original = grid_search.predict(X_test)
    print("Classification Report on Original Test Set:")
    class_report = classification_report(y_test, y_pred_original)
    print(class_report)
    y_pred_prob, y_test = plot_roc_curve_svm(y_test, grid_search, X_test)
    plt.show()
    plot_confusion_matrix(y_test, y_pred_original)
    plt.show()
    
    return y_pred_prob, y_test, class_report, grid_search
