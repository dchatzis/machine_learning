import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 55  ## You will pass it to every sklearn call so we ensure reproducibility

## Load Data from CSV file
df = pd.read_csv("Heart Prediction Quantum Dataset.csv")
#print(df.head())

## Remove target variable
target = df['HeartDisease']
df = df.drop(['HeartDisease', 'QuantumPatternFeature'], axis=1)
#print(df.head()) # last version of data frame
#print(target) # target variable, heart disease
print(len(target))  # length of target variable, heart disease

## Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df, target, train_size=0.8, random_state=RANDOM_STATE)

print(f'train samples: {len(X_train)}\ntest samples: {len(X_test)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')

## A DECISION TREE
min_samples_split_list = list(
    range(2, 25))  ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = list(range(1, 16))

accuracy_list_train = []
accuracy_list_test = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    DTC_model = DecisionTreeClassifier(
        min_samples_split=min_samples_split, random_state=RANDOM_STATE).fit(X_train, y_train)
    predictions_train = DTC_model.predict(X_train)  ## The predicted values for the train dataset
    predictions_test = DTC_model.predict(X_test)  ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_test = accuracy_score(predictions_test, y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train', 'Test'])
plt.show()

accuracy_list_train = []
accuracy_list_test = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    DTC_model = DecisionTreeClassifier(
        max_depth=max_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    predictions_train = DTC_model.predict(X_train)  ## The predicted values for the train dataset
    predictions_test = DTC_model.predict(X_test)  ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_test = accuracy_score(predictions_test, y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train', 'Test'])
plt.show()

decision_tree_model = DecisionTreeClassifier(
    min_samples_split=9, max_depth=4, random_state=RANDOM_STATE).fit(X_train, y_train)

print(
    f"DECISION TREE Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_test),y_test):.4f}"
)

## B RANDOM FOREST
min_samples_split_list = list(
    range(2, 25))  ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = list(range(1, 12))
n_estimators_list = list(range(1, 25))

accuracy_list_train = []
accuracy_list_test = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    RF_model = RandomForestClassifier(
        min_samples_split=min_samples_split, random_state=RANDOM_STATE).fit(X_train, y_train)
    predictions_train = RF_model.predict(X_train)  ## The predicted values for the train dataset
    predictions_test = RF_model.predict(X_test)  ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_test = accuracy_score(predictions_test, y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train', 'Test'])
plt.show()

accuracy_list_train = []
accuracy_list_test = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    RF_model = RandomForestClassifier(
        max_depth=max_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
    predictions_train = RF_model.predict(X_train)  ## The predicted values for the train dataset
    predictions_test = RF_model.predict(X_test)  ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_test = accuracy_score(predictions_test, y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train', 'Test'])
plt.show()

accuracy_list_train = []
accuracy_list_test = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    RF_model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=RANDOM_STATE).fit(X_train, y_train)
    predictions_train = RF_model.predict(X_train)  ## The predicted values for the train dataset
    predictions_test = RF_model.predict(X_test)  ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_test = accuracy_score(predictions_test, y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(n_estimators_list)), labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train', 'Test'])
plt.show()

random_forest_model = RandomForestClassifier(
    n_estimators=21, max_depth=7, min_samples_split=14).fit(X_train, y_train)

print(
    f"RANDOM FOREST Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_test),y_test):.4f}"
)

# Now use gridSearchCV to find best params

# Define parameter grid (these are the hyperparameters to tune)
param_grid = {
    'n_estimators': n_estimators_list,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list
}

# Set up GridSearchCV
if 1 == 0:
    grid_search = GridSearchCV(
        estimator=RF_model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1)  # Use all CPU cores

    # Fit on training data
    grid_search.fit(X_train, y_train)

    # Best model and params
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated score:", grid_search.best_score_)

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Test set accuracy:", accuracy_score(y_test, y_pred))

## C XGBOOST
if 1 == 1:
    n = int(len(X_train) * 0.8)  ## Let's use 80% to train and 20% to eval

    X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[
        n:], y_train[:n], y_train[n:]

    print("y_train_fit type:", type(y_train_fit))
    print("y_train_fit dtype:", getattr(y_train_fit, 'dtype', 'no dtype'))
    print("y_train_fit shape:", np.shape(y_train_fit))

    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        verbosity=1,
        random_state=RANDOM_STATE,
        eval_metric='logloss')  # optional but often needed)

    xgb_model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_train_eval, y_train_eval)],
        early_stopping_rounds=50,
        verbose=True)
    # Here we must pass a list to the eval_set, because you can have several different tuples ov eval sets. The parameter
    # early_stopping_rounds is the number of iterations that it will wait to check if the cost function decreased or not.
    # If not, it will stop and get the iteration that returned the lowest metric on the eval set.

    #xgb_model.best_iteration

    print(
        f"XGBOOST Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_test),y_test):.4f}"
    )
