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

def train_decision_tree_model(df):
    # Splitting the dataset into features and target variable
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Creating the decision tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Defining the hyperparameter grid
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': np.arange(2, 51, 2),
        'min_samples_leaf': np.arange(1, 9),
        'max_depth': [None, 5, 10, 15, 20]
    }

    # Using RandomizedSearchCV for efficient hyperparameter search
    random_search = RandomizedSearchCV(tree_classifier, param_distributions=param_dist, n_iter=100, cv=10, scoring='roc_auc', random_state=42)

    # Measuring the start time for hyperparameter search
    start_time_hyperparameter_search = time()
    random_search.fit(X_train, y_train)
    end_time_hyperparameter_search = time()
    hyperparameter_search_time = end_time_hyperparameter_search - start_time_hyperparameter_search

    # Getting the best parameters and best estimator
    best_params = random_search.best_params_
    best_tree_classifier = random_search.best_estimator_

    # Printing the best parameters and their ROC AUC score on the training set
    y_train_pred_prob = best_tree_classifier.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_train_pred_prob)

    # Training the model on the entire training set and measuring training time
    start_time_training = time()
    best_tree_classifier.fit(X_train, y_train)
    end_time_training = time()
    dt_training_time = end_time_training - start_time_training

    # Predictions on the training and test sets
    y_train_pred = best_tree_classifier.predict(X_train)
    y_test_pred = best_tree_classifier.predict(X_test)

    # Printing performance on the training set
    print("\nPrestazioni sul Set di Addestramento:")
    print(classification_report(y_train, y_train_pred))

    # Printing performance on the test set
    print("\nPrestazioni sul Set di Test:")
    print(classification_report(y_test, y_test_pred))

    # Printing the best parameters and time taken for hyperparameter search and training
    print("\nMigliori Parametri:", best_params)
    print("Tempo impiegato per la Ricerca degli Iperparametri:", hyperparameter_search_time, "secondi")
    print("Tempo impiegato per l'Addestramento:", dt_training_time, "secondi")
    
    plot_decision_tree(random_search.best_estimator_, feature_names=X_train.columns)
    plot_feature_importance_decision_tree(best_tree_classifier, X)
    plot_roc_curve(y_test, best_tree_classifier, X_test)
    plot_confusion_matrix(y_test, y_test_pred)
    
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
    
    
def k_fold_cross_validation_dt(model, df, n_fold=10):
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    folds = KFold(n_splits=n_fold, shuffle=True)

    accuracy_k_fold_dt = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        # Make predictions on the validation set and calculate accuracy
        y_valid_pred = model.predict(X_valid_fold)
        accuracy_k_fold_dt.append(accuracy_score(y_valid_fold, y_valid_pred))

    print("Accuracy for each fold:", accuracy_k_fold_dt)
    print("Mean accuracy:", np.mean(accuracy_k_fold_dt))
