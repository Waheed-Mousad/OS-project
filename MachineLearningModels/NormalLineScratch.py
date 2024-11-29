import pandas as pd
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from BatterSplit import batterSplit
import numpy as np
import time

class lineatRegressionScratch:
    """
    Linear regression from scratch model, have train and predict methods
    train method:
    it takes X_train, y_train, X_test, y_test dataframes as inputs
    during training, it will print the MAE for training and testing data
    predict method:
    it takes X_test  dataframe as input and return the predicted values in new column
    named after the target column saved during training as return the dataframe
    """
    def __init__(self):
        self.weights = None
        self.feature_names = None
        self.target_col = None
        self.mae_train = None
        self.mae_test = None
        self.feature_mapping = {}
    def train(self, X_train, y_train, X_test, y_test):
        # copy the dataframe to not effect the original dataframe
        X_train = X_train.copy()
        X_test = X_test.copy()
        self.target_col = y_train.columns[0]
        # save the columns names
        temp_names = ['bias']
        for i in X_train.columns:
            temp_names.append(i)
        # Add a column of ones to X_train and X_test
        X_train.insert(0, 'Ones', 1)
        X_test.insert(0, 'Ones', 1)
        # Convert X_train, X_test, y_train, y_test to numpy arrays
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        # calculate weights using normal equation
        # b = (XT*X)^-1 * XT * Y
        X_train_T = np.transpose(X_train)  # Transpose of X_train
        X_train_T_X_train = np.dot(X_train_T, X_train)  # X_train_T * X_train
        X_train_T_X_train_inv = np.linalg.inv(X_train_T_X_train)  # Inverse of X_train_T_X_train
        X_train_T_Y = np.dot(X_train_T, y_train)  # X_train_T * Y
        b = np.dot(X_train_T_X_train_inv, X_train_T_Y)  # Final weights
        # save the weights in the dictionary
        for i in range(len(temp_names)):
            self.feature_mapping[temp_names[i]] = b[i]
        # calculate MAE for training data and testing data
        y_train_pred = np.dot(X_train, b)
        y_test_pred = np.dot(X_test, b)
        mae_train = np.mean(np.abs(y_train - y_train_pred))
        mae_test = np.mean(np.abs(y_test - y_test_pred))
        YELLOW = "\033[33m"  # Yellow color
        RED = "\033[31m"  # Red color
        RESET = "\033[0m"  # Reset to default color
        print(
            f"{YELLOW}(Linear regression using normal line from scratch){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
        print(
            f"{YELLOW}(Linear regression using normal line from scratch){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
        self.weights = b
        self.mae_train = mae_train
        self.mae_test = mae_test

        return mae_train,mae_test

    def predict(self, X):
        # copy the dataframe to not effect the original dataframe
        X = X.copy()
        # check if bias column is in the dataframe
        if 'bias' not in X.columns:
            X.insert(0, 'bias', 1)
        y_pred = np.dot(X, self.weights)
        X[self.target_col] = y_pred
        # remove the bias column
        X.drop('bias', axis=1, inplace=True)
        return X

if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    X_train, X_test, y_train, y_test = batterSplit(df, 'RunTime ', 0.2)
    model = lineatRegressionScratch()
    model.train(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    # calculate mae
    mae_test = np.mean(np.abs(y_test - y_pred))
    # if the value here match the one in the class, then the prediction method is working correctly
    print(f"MAE for testing data: {mae_test}")
    print(y_pred)

