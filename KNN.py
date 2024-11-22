import pandas as pd
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def KNN_lib(X_train, y_train, X_test, y_test):
    # Create a KNN model
    model = KNeighborsRegressor(n_neighbors=5)
    # Fit the model
    model.fit(X_train, y_train)
    # Predict the data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Calculate MAE
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    YELLOW = "\033[33m"  # Yellow color
    RED = "\033[31m"  # Red color
    RESET = "\033[0m"  # Reset to default color
    print(
        f"{YELLOW}(KNN from library){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
    print(
        f"{YELLOW}(KNN from library){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
    return

if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    X = df.drop('RunTime ', axis=1)
    y = df['RunTime ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    KNN(X_train, y_train, X_test, y_test)