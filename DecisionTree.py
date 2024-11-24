import pandas as pd
import matplotlib.pyplot as plt
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import time
def DecisionTree_lib(X_train, y_train, X_test, y_test):
    # Create a DecisionTree model
    model = DecisionTreeRegressor()
    # Fit the model
    model.fit(X_train, y_train)
    # Predict the data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Calculate MAE
    mae_train = mean_absolute_error(y_train, y_train_pred)
    start = time.time()
    mae_test = mean_absolute_error(y_test, y_test_pred)
    end = time.time()

    # Get feature importances
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    # Sort features by importance in descending order
    sorted_indices = feature_importances.argsort()[::-1]  # Indices of sorted importances (descending)
    sorted_features = feature_names[sorted_indices]       # Sorted feature names
    sorted_importances = feature_importances[sorted_indices]  # Sorted importances

    YELLOW = "\033[33m"  # Yellow color
    RED = "\033[31m"  # Red color
    RESET = "\033[0m"  # Reset to default color
    BLUE = "\033[34m"  # Blue color
    GREEN = "\033[32m"  # Green color
    # print most important features first
    print(f"{RED}How important is each feature according to decision tree?{RESET}:")
    for i in range(len(model.feature_importances_)):
        print(f"{GREEN}{X_train.columns[i]}{RESET}: {BLUE}{model.feature_importances_[i]}{RESET}")
    print(
        f"{YELLOW}(Decision Tree from library){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
    print(
        f"{YELLOW}(Decision Tree from library){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
    print(f"{YELLOW}Time taken to predict the testing data:{RESET} {RED}{end - start} seconds{RESET}")

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importances, color='skyblue')
    plt.title('Feature Importances in Decision Tree')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    #plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
    plt.tight_layout()
    plt.show()

    return mae_train,mae_test



if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    X = df.drop('RunTime ', axis=1)
    y = df['RunTime ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    DecisionTree_lib(X_train, y_train, X_test, y_test)
