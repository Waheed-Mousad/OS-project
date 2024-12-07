import pandas as pd
from MachineLearningModels.DataPreProcess import data_preprocess
from MachineLearningModels.BatterSplit import batterSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import time
class LinearRegression_lib:
    """
    Linear regression from library model, have train and predict methods
    train method:
    it takes X_train, y_train, X_test, y_test dataframes as inputs
    during training, it will print the MAE for training and testing data
    predict method:
    it takes X_test  dataframe as input and return the predicted values in new column
    named after the target column saved during training as return the dataframe
    """
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.target_col = None
        self.mae_train = None
        self.mae_test = None
    def train(self, X_train, y_train, X_test, y_test):
        # Create a linear regression model
        model = LinearRegression()
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
        YELLOW = "\033[33m"  # Yellow color
        RED = "\033[31m"  # Red color
        RESET = "\033[0m"  # Reset to default color
        print(
            f"{YELLOW}(Linear regression from library){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
        print(
            f"{YELLOW}(Linear regression from library){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
        print(f"{YELLOW}Time taken to predict the testing data:{RESET} {RED}{end - start} seconds{RESET}")
        self.model = model
        self.mae_train = mae_train
        self.mae_test = mae_test
        self.target_col = y_train.columns[0]
        self.feature_names = X_train
        return mae_train,mae_test

    def predict(self, x):
        # copy the dataframe to not effect the original dataframe
        X = x.copy()
        y_pred = self.model.predict(X)
        X[self.target_col] = y_pred
        return X


if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    X_train, X_test, y_train, y_test = batterSplit(df, 'RunTime ', 0.2)
    model = linearRegression_lib()
    model.train(X_train, y_train, X_test, y_test)
    print(model.predict(X_test))
