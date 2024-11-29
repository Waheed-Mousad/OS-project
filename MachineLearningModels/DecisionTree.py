import pandas as pd
import matplotlib.pyplot as plt
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import time
from BatterSplit import batterSplit
class DecisionTree:
    """
    Decision Tree model from library, have train and predict methods
    train method:
    it takes X_train, y_train, X_test, y_test dataframes as inputs
    during training, it will print the most important features in the dataset
    as well as the MAE for training and testing data
    predict method:
    it takes X_test  dataframe as input and return the predicted valuesin new column
    names after the target column saved during training as return the dataframe
    plot_feature_importances method:
    plot the feature importances in descending order
    """
    def __init__(self):
        self.model = None
        self.feature_importances = None
        self.feature_names = None
        self.target_col = None
        self.mae_train = None
        self.mae_test = None
    def train(self, X_train, y_train, X_test, y_test):
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
        self.model = model
        self.feature_importances = feature_importances
        self.feature_names = feature_names
        self.mae_train = mae_train
        self.mae_test = mae_test
        self.target_col = y_train.columns[0]

    def predict(self, X):
        y_pred = self.model.predict(X)
        X[self.target_col] = y_pred
        return X

    def plot_feature_importances(self):
        # Sort features by importance in descending order
        sorted_indices = self.feature_importances.argsort()[::-1]
        sorted_features = self.feature_names[sorted_indices]
        sorted_importances = self.feature_importances[sorted_indices]

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_features, sorted_importances, color='skyblue')
        plt.title('Feature Importances in Decision Tree')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        # plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    # split the data
    X_train, X_test, y_train, y_test = batterSplit(df, 'RunTime ')
    dec_tree = DecisionTree()
    dec_tree.train(X_train, y_train, X_test, y_test)
    predicted_df = dec_tree.predict(X_test)
    print(predicted_df.head())
    print(f"Feature importance's: {dec_tree.target_col}")
    dec_tree.plot_feature_importances()

