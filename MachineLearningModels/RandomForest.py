import pandas as pd
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from BatterSplit import batterSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import time
class RandomForest_lib:
    """
    Random Forest model from library, have train and predict methods
    train method:
    it takes X_train, y_train, X_test, y_test dataframes as inputs
    during training, it will print the most important features in the dataset
    as well as the MAE for training and testing data
    predict method:
    it takes X_test  dataframe as input and return the predicted values in new column
    named after the target column saved during training as return the dataframe
    """
    def __init__(self):
        self.model = None
        self.feature_importances = None
        self.feature_names = None
        self.target_col = None
        self.mae_train = None
        self.mae_test = None
    def train(self, X_train, y_train, X_test, y_test):
        # Create a RandomForest model
        model = RandomForestRegressor(random_state=42)
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
        BLUE = "\033[34m"  # Blue color
        GREEN = "\033[32m"  # Green color
        # Print most important features first
        print(f"{RED}How important is each feature according to the random forest?{RESET}:")
        for i in range(len(model.feature_importances_)):
            print(f"{GREEN}{X_train.columns[i]}{RESET}: {BLUE}{model.feature_importances_[i]}{RESET}")
        print(
            f"{YELLOW}(Random Forest from library){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
        print(
            f"{YELLOW}(Random Forest from library){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
        print(f"{YELLOW}Time taken to predict the testing data:{RESET} {RED}{end - start} seconds{RESET}")
        self.model = model
        self.feature_importances = model.feature_importances_
        self.feature_names = X_train.columns
        self.mae_train = mae_train
        self.mae_test = mae_test
        self.target_col = y_train.columns[0]
        return mae_train,mae_test
    def predict(self, X):
        y_pred = self.model.predict(X)
        X[self.target_col] = y_pred
        return X



if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    # Preprocess the data
    df, mapping = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    # Split the data
    X_train, X_test, y_train, y_test = batterSplit(df, 'RunTime ', 0.2)
    # Create a RandomForest model
    model = RandomForest_lib()
    model.train(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    print(y_pred)
