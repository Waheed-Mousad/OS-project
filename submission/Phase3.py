import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
def batterSplit(df, target_col='', test_size=0.2):
    """
    Split the data into training and testing dataframes. instead of numpy arrays
    """
    X = df.drop(target_col, axis=1)
    y = df[[target_col]]  # This keeps 'RunTime ' as a DataFrame
    # save the columns names
    feature_names = X.columns
    target_col = y.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # turn X_train, X_test, y_train, y_test back to dataframes
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.DataFrame(y_train, columns=target_col)
    y_test = pd.DataFrame(y_test, columns=target_col)
    return X_train, X_test, y_train, y_test

def data_preprocess(df, target_col=[], samples=0, no_norm_col=[]):
    if len(target_col) != 0:
        df = df[target_col]
    # seperate the column that should not be normalized
    if len(no_norm_col) != 0:
        for col in no_norm_col:
            temp_df = df[col]
            df = df.drop(col, axis=1)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Remove rows where any numeric column contains 0 or negative values
    df = df[(df[numeric_cols] > 0).all(axis=1)]
    # map the catagorical column and save the mapping
    catagorical_cols = df.select_dtypes(include=['object']).columns
    catagorical_mapping = {}
    for col in catagorical_cols:
        df[col], catagorical_mapping[col] = pd.factorize(df[col])
    # Normalize the numeric columns and save min-max values
    numeric_mapping = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
        numeric_mapping[col] = (min_val, max_val)
    # Save the mapping
    mapping = {'catagorical': catagorical_mapping, 'numeric': numeric_mapping}
    # Add the columns that should not be normalized at the beggining of original dataframe
    if len(no_norm_col) != 0:
        for col in no_norm_col:
            df.insert(0, col, temp_df)
    # Sample the data
    if samples != 0:
        df = df.sample(n=samples, random_state=42)
    return df, mapping

def denormlize_data(df, mapping, no_norm_col=[]):
    # ignore the columns that did not get normalized
    if len(no_norm_col) != 0:
        for col in no_norm_col:
            temp_df = df[col]
            df = df.drop(col, axis=1)
    # Denormalize numeric columns
    for col in mapping['numeric']:
        df[col] = df[col] * (mapping['numeric'][col][1] - mapping['numeric'][col][0]) + mapping['numeric'][col][0]
    # Denormalize categorical columns
    for col in mapping['catagorical']:
        # Reverse the mapping, turn into integer before mapping
        df[col] = df[col].apply(lambda x: int(x))
        df[col] = df[col].apply(lambda x: mapping['catagorical'][col][x])
    # add the columns that did not get normalized
    if len(no_norm_col) != 0:
        for col in no_norm_col:
            df.insert(0, col, temp_df)
    return df

class LineatRegressionScratch:
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
        start = time.time()
        y_test_pred = np.dot(X_test, b)
        end = time.time()
        mae_train = np.mean(np.abs(y_train - y_train_pred))
        mae_test = np.mean(np.abs(y_test - y_test_pred))
        YELLOW = "\033[33m"  # Yellow color
        RED = "\033[31m"  # Red color
        RESET = "\033[0m"  # Reset to default color
        print(
            f"{YELLOW}(Linear regression using normal line from scratch){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
        print(
            f"{YELLOW}(Linear regression using normal line from scratch){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
        print(f"{YELLOW}Time taken to predict the testing data:{RESET} {RED}{end - start} seconds{RESET}")
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

def FCFS(df, burst_time_col=''):
    """
    First Come First Serve Algorithm
    input: dataframe, burst_time_col,
    output: same data frame with new columns: waiting_time, turn_around_time as well as average waiting time and average turn around time
    """
    #assuming the data arrived at time 0
    df['FCFS waiting_time'] = 0
    df['FCFS turn_around_time'] = 0
    # calculate waiting time and turn around time for each process
    for idx in df.index:
        if idx == df.index[0]:
            df.loc[idx, 'FCFS waiting_time'] = 0  # First process has no waiting time
        else:
            prev_idx = df.index[df.index.get_loc(idx) - 1]  # Get the previous index
            df.loc[idx, 'FCFS waiting_time'] = df.loc[prev_idx, burst_time_col] + df.loc[prev_idx, 'FCFS waiting_time']

        df.loc[idx, 'FCFS turn_around_time'] = df.loc[idx, burst_time_col] + df.loc[idx, 'FCFS waiting_time']
    # calculate the average waiting time and average turn around time
    avg_waiting_time = df['FCFS waiting_time'].mean()
    avg_turn_around_time = df['FCFS turn_around_time'].mean()
    return df, avg_waiting_time, avg_turn_around_time

def SJF(df, burst_time_col=''):
    """
    Shortest Job First Algorithm
    input: dataframe, burst_time_col
    output: same data frame with new columns: waiting_time, turn_around_time as well as average waiting time and average turn around time
    """
    # sort the data frame based on burst time
    df = df.sort_values(by=burst_time_col)
    #assuming the data arrived at time 0
    df['SJF waiting_time'] = 0
    df['SJF turn_around_time'] = 0
    # calculate waiting time and turn around time for each process
    for idx in df.index:
        if idx == df.index[0]:
            df.loc[idx, 'SJF waiting_time'] = 0  # First process has no waiting time
        else:
            prev_idx = df.index[df.index.get_loc(idx) - 1]  # Get the previous index
            df.loc[idx, 'SJF waiting_time'] = df.loc[prev_idx, burst_time_col] + df.loc[prev_idx, 'SJF waiting_time']

        df.loc[idx, 'SJF turn_around_time'] = df.loc[idx, burst_time_col] + df.loc[idx, 'SJF waiting_time']
    # calculate the average waiting time and average turn around time
    avg_waiting_time = df['SJF waiting_time'].mean()
    avg_turn_around_time = df['SJF turn_around_time'].mean()
    return df, avg_waiting_time, avg_turn_around_time

def plot_turn_around_vs_waiting_time(package, extra_info = ''):
    df1 = package[0][0].copy()
    df2 = package[1][0].copy()
    avg_waiting_time1 = package[0][1]
    avg_turn_around_time1 = package[0][2]
    avg_waiting_time2 = package[1][1]
    avg_turn_around_time2 = package[1][2]
    # re index the data frame make sure they start from 0, the real index is not 0
    df1.index = range(1, len(df1) + 1)
    df2.index = range(1, len(df2) + 1)
    # plot waiting time only in Y axis and JobID in X axis , take job ID = index starting from 1
    plt.plot(df1.index, df1['SJF waiting_time'], label='SJF waiting time', marker='o', linestyle='-')
    plt.plot(df2.index, df2['FCFS waiting_time'], label='FCFS waiting time', marker='o', linestyle='-')
    #print the average waiting time of each algorithm on the side of the plot
    plt.text(0, avg_waiting_time1, f'Avg Waiting Time SJF: {avg_waiting_time1}', fontsize=12, color='blue')
    plt.text(0, avg_waiting_time2, f'Avg Waiting Time FCFS: {avg_waiting_time2}', fontsize=12, color='orange')
    plt.xlabel('JobID')
    plt.ylabel('Time as unit of time')
    title = 'Waiting Time SJF vs FCFS'
    if len(extra_info) != 0:
        title = title + ' for ' + extra_info
    plt.title(title)
    plt.legend()
    plt.show()
    # plot turn around time only in Y axis and JobID in X axis , take job ID = index starting from 1
    plt.plot(df1.index, df1['SJF turn_around_time'], label='SJF turn around time', marker='o', linestyle='-')
    plt.plot(df2.index, df2['FCFS turn_around_time'], label='FCFS turn around time', marker='o', linestyle='-')
    #print the average turn around time of each algorithm on the side of the plot
    plt.text(0, avg_turn_around_time1, f'Avg Turn Around Time SJF: {avg_turn_around_time1}', fontsize=12, color='blue')
    plt.text(0, avg_turn_around_time2, f'Avg Turn Around Time FCFS: {avg_turn_around_time2}', fontsize=12, color='orange')
    plt.xlabel('JobID')
    plt.ylabel('Time as unit of time')
    title = 'Turn Around Time SJF vs FCFS'
    if len(extra_info) != 0:
        title = title + ' for ' + extra_info
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # load the two dataset and preprocess them
    df = pd.read_csv('processes_datasets.csv')
    df1, mapping1 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    df2, mapping2 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=1000)
    # split the two datasets
    X_train1, X_test1, y_train1, y_test1 = batterSplit(df1, 'RunTime ', 0.2)
    X_train2, X_test2, y_train2, y_test2 = batterSplit(df2, 'RunTime ', 0.2)
    # define the two models for dataset 1, linear regression from scratch and for dataset 2, linear regression from library
    linear_scratch = LineatRegressionScratch()
    linear_lib = LinearRegression_lib()
    # train the two models on their datasets
    linear_scratch.train(X_train1, y_train1, X_test1, y_test1)
    linear_lib.train(X_train2, y_train2, X_test2, y_test2)
    # create a prediction dataframes for the two models
    y_pred_scratch = linear_scratch.predict(X_test1)
    y_pred_lib = linear_lib.predict(X_test2)
    # denormalize the dataframes
    y_pred_scratch = denormlize_data(y_pred_scratch, mapping1)
    y_pred_lib = denormlize_data(y_pred_lib, mapping2)
    # Schedule the dataframes
    df_scratch1, avg_waiting_time_scratch1, avg_turn_around_time_scratch1 = SJF(y_pred_scratch, 'RunTime ')
    df_scratch2, avg_waiting_time_scratch2, avg_turn_around_time_scratch2 = FCFS(y_pred_scratch, 'RunTime ')

    df_lib1, avg_waiting_time_lib1, avg_turn_around_time_lib1 = SJF(y_pred_lib, 'RunTime ')
    df_lib2, avg_waiting_time_lib2, avg_turn_around_time_lib2 = FCFS(y_pred_lib, 'RunTime ')

    # PACKAGE the data to pass them to the plot function
    package1 = [[df_scratch1, avg_waiting_time_scratch1, avg_turn_around_time_scratch1], [df_scratch2, avg_waiting_time_scratch2, avg_turn_around_time_scratch2]]
    package2 = [[df_lib1, avg_waiting_time_lib1, avg_turn_around_time_lib1], [df_lib2, avg_waiting_time_lib2, avg_turn_around_time_lib2]]

    # PLOT
    plot_turn_around_vs_waiting_time(package1, "linear regression from scratch")
    plot_turn_around_vs_waiting_time(package2, "linear regression from library")





