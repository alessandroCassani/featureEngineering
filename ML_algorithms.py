from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from time import time
import numpy as np
from sklearn.tree import plot_tree

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














def plot_decision_tree(tree_model, feature_names, class_names=['0', '1']):
    plt.figure(figsize=(20, 10))
    tree_plot = plot_tree(tree_model, filled=True, feature_names=feature_names, class_names=class_names, rounded=True, impurity=False, fontsize=8)
    # Get the labels of each node and display them
    text = tree_plot[0]
    print("Node Labels:\n", text)
    plt.show()
