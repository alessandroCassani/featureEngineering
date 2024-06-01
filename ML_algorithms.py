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
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import randint
from sklearn.model_selection import learning_curve
import scipy.stats as st  

def train_decision_tree_model(df_dirty, df_original):
    # Splitting the dataset con duplicati into features and target variable
    X = df_dirty.drop('stroke', axis=1)
    y = df_dirty['stroke']

    # Splitting the dataset originale into features and target variable
    X_test_original = df_original.drop('stroke', axis=1)
    y_test_original = df_original['stroke']

    # Splitting the dirty dataset into training set and test set (with 30% testing)
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X, y, test_size=0.3, random_state=42)

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

    # Predictions on test sets
    y_train_pred_dirty = best_tree_classifier.predict(X_train_dirty)
    y_test_pred_dirty = best_tree_classifier.predict(X_test_dirty)
    y_test_pred_original = best_tree_classifier.predict(X_test_original)

    # Valutazione delle prestazioni sul set di addestramento
    print("Classification Report on Training Set:")
    print(classification_report(y_train_dirty, y_train_pred_dirty))

    # Printing performance on the test set dirty
    print("Classification Report on Test Set - dirty:")
    print(classification_report(y_test_dirty, y_test_pred_dirty))

    # Printing performance on the test set original
    print("Classification Report on Test Set - original:")
    print(classification_report(y_test_original, y_test_pred_original))

    # Printing the best parameters and time taken for hyperparameter search and training
    print("\nMigliori Parametri:", best_params)
    print("Tempo impiegato per la Ricerca degli Iperparametri:", hyperparameter_search_time, "secondi")
    print("Tempo impiegato per l'Addestramento:", dt_training_time, "secondi")
    
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

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
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


def train_hist_gradient_boosting_model(df_dirty, df_original):
    # Splitting the dataset con duplicati into features and target variable
    X = df_dirty.drop('stroke', axis=1)
    y = df_dirty['stroke']

    # Splitting the dataset originale into features and target variable
    X_test_original = df_original.drop('stroke', axis=1)
    y_test_original = df_original['stroke']

    # Splitting the dataset into training set and test set (with 30% testing)
    X_train_dirty, X_test_dirty, y_train_dirty, y_test_dirty = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the parameter grid
    param_grid = {
        'max_iter': randint(50, 500),
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': randint(3, 10),
        'min_samples_leaf': randint(1, 20),
        'l2_regularization': [0.0, 0.1, 0.2, 0.3],
    }

    # Create a HistGradientBoostingClassifier
    hgb_classifier = HistGradientBoostingClassifier()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        hgb_classifier,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring='recall',
        verbose=2,
        n_jobs=-1
    )

    start_time = time()
    random_search.fit(X_train_dirty, y_train_dirty)
    end_time = time()
    search_time = end_time - start_time
    
    print(f'search time: {search_time}')

    # Print the best parameters
    print("Best parameters found: ", random_search.best_params_)

    # Print the best score on training data
    print("Best score on training data: ", random_search.best_score_)

    # Get the best model
    best_model = random_search.best_estimator_

    # Predictions on training set
    train_predictions = best_model.predict(X_train_dirty)

    # Print classification report for training set
    print("Classification Report on Training Set - dirty:")
    print(classification_report(y_train_dirty, train_predictions))

    # Predictions on test set dirty
    test_predictions_dirty = best_model.predict(X_test_dirty)

    # Print classification report for test set
    print("Classification Report on Test Set - dirty:")
    print(classification_report(y_test_dirty, test_predictions_dirty))
    
    # Predictions on test set original
    test_predictions_original = best_model.predict(X_test_original)

    # Print classification report for test set original
    print("Classification Report on Test Set - original:")
    print(classification_report(y_test_original, test_predictions_original))
    

    plot_roc_curve(y_test_original, best_model, X_test_original)
    plot_confusion_matrix(y_test_original, test_predictions_original)
    plot_learning_curve(HistGradientBoostingClassifier(), X_train_dirty, y_train_dirty, cv=5)

    return best_model

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.title('Learning Curve (HistGradientBoosting)')
    plt.show()
    