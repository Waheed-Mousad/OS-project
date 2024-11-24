import pandas as pd
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from sklearn.model_selection import train_test_split
from NormalLineScratch import NormalLine
from linearRegressionlib import NormalLine_lib
from NurealNetwork import N_network
from KNN import KNN_lib
from SVR import SVR_lib
from DecisionTree import DecisionTree_lib


if __name__ == '__main__':
    df = pd.read_csv('Original_processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    #data_visualization(df, "RunTime ")
    X = df.drop('RunTime ', axis=1)
    y = df['RunTime ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    GREEN = '\033[32m'
    RESET = '\033[0m'
    print(f"{GREEN}############### Decision Tree from library ##############{RESET}")
    DecisionTree_lib(X_train, y_train, X_test, y_test)
    print(f"{GREEN}############### Normal Line from scratch ################{RESET}")
    NormalLine(X_train, y_train, X_test, y_test)
    print(f"{GREEN}############### Normal Line from library ################{RESET}")
    NormalLine_lib(X_train, y_train, X_test, y_test)
    print(f"{GREEN}############### Neural Network from library #############{RESET}")
    N_network(X_train, X_test, y_train, y_test)
    print(f"{GREEN}############### KNN from library ########################{RESET}")
    KNN_lib(X_train, y_train, X_test, y_test)
    print(f"{GREEN}############### SVR from library ########################{RESET}")
    SVR_lib(X_train, y_train, X_test, y_test)