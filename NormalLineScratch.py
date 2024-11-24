import pandas as pd
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from sklearn.model_selection import train_test_split
import numpy as np
def NormalLine(X_train, y_train, X_test, y_test):
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
    return

if __name__ == '__main__':
    df = pd.read_csv('Modified_Process_dataset.csv')
    X = df.drop('RunTime ', axis=1)
    y = df['RunTime ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    NormalLine(X_train, y_train, X_test, y_test)
