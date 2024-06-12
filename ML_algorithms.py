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


def train_decision_tree_model(df_dirty, df_original):
    # Splitting the dataset con duplicati into features and target variable
    X_dirty = df_dirty.drop('stroke', axis=1)
    y_dirty = df_dirty['stroke']
    
    X_original = df_original.drop('stroke', axis=1)
    y_original = df_original['stroke']

    # Splitting the dirty dataset into training set and test set (with 30% testing)
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=42)
    
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

    # Creating the decision tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Defining the hyperparameter grid
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': np.arange(2, 51, 2),
        'min_samples_leaf': np.arange(1, 9),
        'max_depth': [None, 5, 10, 15]
    }

    # Using RandomizedSearchCV for efficient hyperparameter search
    random_search = RandomizedSearchCV(tree_classifier, param_distributions=param_dist, n_iter=100, cv=10, scoring='roc_auc', random_state=42)

    # Measuring the start time for hyperparameter search
    start_time_hyperparameter_search = time()
    random_search.fit(X_train_dirty, y_train_dirty)
    end_time_hyperparameter_search = time()
    hyperparameter_search_time = end_time_hyperparameter_search - start_time_hyperparameter_search

    # Getting the best parameters and best estimator
    best_params = random_search.best_params_
    best_tree_classifier = random_search.best_estimator_

    # Training the model on the entire training set and measuring training time
    start_time_training = time()
    best_tree_classifier.fit(X_train_dirty, y_train_dirty)
    end_time_training = time()
    dt_training_time = end_time_training - start_time_training

    # Predictions 
    y_train_pred_dirty = best_tree_classifier.predict(X_train_dirty)
    y_test_pred_original = best_tree_classifier.predict(X_test_original)

    # Valutazione delle prestazioni sul set di addestramento
    print("Classification Report on Training Set:")
    print(classification_report(y_train_dirty, y_train_pred_dirty))

    # Printing performance on the test set original
    print("Classification Report on Test Set - original:")
    print(classification_report(y_test_original, y_test_pred_original))

    # Printing the best parameters and time taken for hyperparameter search and training
    print("\nbest hyperparameter:", best_params)
    print("hyperparameter's time search:", hyperparameter_search_time, "seconds")
    print("training time needed:", dt_training_time, "seconds")
    
    plot_decision_tree(random_search.best_estimator_, feature_names=X_train_dirty.columns)
    plot_feature_importance_decision_tree(best_tree_classifier, X_train_dirty)
    plot_roc_curve(y_test_original, best_tree_classifier, X_test_original)
    plot_confusion_matrix(y_test_original, y_test_pred_original)

    return best_tree_classifier

def plot_decision_tree(tree_model, feature_names, class_names=['0', '1']):
    plt.figure(figsize=(20, 10))
    tree_plot = plot_tree(tree_model, filled=True, feature_names=feature_names, class_names=class_names, rounded=True, impurity=False, fontsize=8)
    # Get the labels of each node and display them
    text = tree_plot[0]
    print("Node Labels:\n", text)
    plt.show()
    
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
    
def plot_confusion_matrix(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    labels = [1, 0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    
    
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
    # Define parameter distributions for randomized search
    param_distributions = {'C': reciprocal(0.1, 1000), 
                           'gamma': reciprocal(0.001, 1),
                           'kernel': ['rbf', 'linear']}
    
    # Split the original dataset into features and target variable
    X_original = df_original.drop('stroke', axis=1)
    y_original = df_original['stroke']
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

    # Split the dirty dataset into features and target variable
    X_dirty = df_dirty.drop('stroke', axis=1)
    y_dirty = df_dirty['stroke']
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=42)
    
    # Initialize SVM model
    svm_model = SVC(kernel='linear', random_state=0)
    svm_model.fit(X_train_dirty, y_train_dirty)

    # Predict on the dirty test set
    y_pred_dirty = svm_model.predict(X_test_dirty)
    
    # Printing performance on the dirty test set
    print("Classification Report on Dirty Test Set:")
    print(classification_report(y_test_dirty, y_pred_dirty))

    # Predict on the original test set
    y_pred_original = svm_model.predict(X_test_original)
    
    # Printing performance on the original test set
    print("Classification Report on Original Test Set:")
    print(classification_report(y_test_original, y_pred_original))
    
    # Plot ROC curve
    plot_roc_curve_svm(y_test_original, svm_model, X_test_original)

    
    # Plot confusion matrix
    plot_confusion_matrix(y_test_original, y_pred_original)
    
    return svm_model


    
def model_dt(df_dirty, df_original):
    # Splitting the dataset con duplicati into features and target variable
    X_dirty = df_dirty.drop('stroke', axis=1)
    y_dirty = df_dirty['stroke']
    
    X_original = df_original.drop('stroke', axis=1)
    y_original = df_original['stroke']

    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

    # Splitting the dirty dataset into training set and test set (with 30% testing)
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X_dirty, y_dirty, test_size=0.3, random_state=42)
    
    decision_tree_model = DecisionTreeClassifier(max_depth=10, random_state=0)
    decision_tree_model.fit(X_train_dirty, y_train_dirty)
    print("\n--- Prestazioni del modello Decision Tree applicato al set di Test: \n")
    y_pred_dirty = decision_tree_model.predict(X_test_dirty)
    y_pred_original = decision_tree_model.predict(X_test_original)
    
    print("Classification Report on Test Set - dirty:")
    print(classification_report(y_test_dirty, y_pred_dirty))

    # Printing performance on the test set original
    print("Classification Report on Test Set - original:")
    print(classification_report(y_test_original, y_pred_original))
    
    plot_decision_tree(decision_tree_model,df_original.columns,)
    plot_feature_importance_decision_tree(decision_tree_model, X_train_dirty)
    plot_roc_curve(y_test_original, decision_tree_model, X_test_original)
    plot_confusion_matrix(y_test_original, y_pred_original)
    return decision_tree_model

def plot_roc_curve_svm(y_test, classifier, X_test):
    y_pred_prob = classifier.decision_function(X_test)
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