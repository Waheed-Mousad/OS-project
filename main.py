"""
PLAN: have 3 functions in main.py
1. train all the models
2. plot models comparison
3. run scheduling algorithm on the model depending on the user input

goals:
have elegent ways to plot :)


"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MachineLearningModels.DataPreProcess import data_preprocess, data_visualization, denormlize_data
from MachineLearningModels.KNN import KNN_lib
from MachineLearningModels.DecisionTree import DecisionTree_lib
from MachineLearningModels.RandomForest import RandomForest_lib
from MachineLearningModels.SVR import SVR_lib
from MachineLearningModels.linearRegressionlib import LinearRegression_lib
from MachineLearningModels.NurealNetwork import N_network
from MachineLearningModels.NormalLineScratch import LineatRegressionScratch
from MachineLearningModels.BatterSplit import batterSplit
import pandas as pd
from SchedulingAlgorithms.FirstComeFirstServe import FCFS
from SchedulingAlgorithms.ShortestJobFirst import SJF


if __name__ == '__main__':
    GREEN = "\033[32m"  # Green color
    RESET = "\033[0m"  # Reset to default color
    RED = "\033[31m"  # Red color
    YELLOW = "\033[33m"  # Yellow color
    CYAN = "\033[36m"  # Cyan color
    MAGENTA = "\033[35m"  # Magenta color
    # Read the data
    df = pd.read_csv(os.path.join('MachineLearningModels','processes_datasets.csv'))
    #Re index so index = JobID
    df.index = df.index + 1
    data1, mapping1 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
         , 'JobID'], samples=100, no_norm_col=['JobID'])
    data2, mapping2 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
        , 'JobID'], samples=1000, no_norm_col=['JobID'])
    print(f"{GREEN}Dataset 1 ready with {RED}{data1.shape[0]}{GREEN} rows{RESET}")
    print(f"{GREEN}Dataset 2 ready with {RED}{data2.shape[0]}{GREEN} rows{RESET}")
    # Split the data
    X_train1, X_test1, y_train1, y_test1 = batterSplit(data1, 'RunTime ', 0.2)
    X_train2, X_test2, y_train2, y_test2 = batterSplit(data2, 'RunTime ', 0.2)
    print(f"{GREEN}both Data splitted with ratio {RED}0.2{RESET}")
    # main program
    while True:
        # ask the user which dataset to work on first
        print(f"{CYAN}Which dataset would you like to work on first?{RESET}")
        print(f"{YELLOW}1{RESET}. {MAGENTA}Dataset 1{RESET}")
        print(f"{YELLOW}2{RESET}. {MAGENTA}Dataset 2{RESET}")
        print(f"{YELLOW}3{RESET}. {RED}Exit{RESET}")
        choice = input()
        if choice == '1':
            print(f"{GREEN}######Working on Dataset 1######{RESET}")
            # TODO ask the user which model to train on or all after that wait for the user confirmation to run the scheduling algorithm on predicted results
            pass
        elif choice == '2':
            print(f"{GREEN}######Working on Dataset 2######{RESET}")
            # TODO ask the user which model to train on or all after that wait for the user confirmation to run the scheduling algorithm on predicted results
            pass
        elif choice == '3':
            print(f"{RED}Exiting...{RESET}")
            break
        else:
            print(f"{RED}Invalid choice{RESET}")
            continue




    # a function to train the model based on the user input based on the dataset the user choose
    def train_model(dataset, model=0):
        """
         TODO DO WHAT THE USER REQUESTED AND WAIT FOR CONFIRMATION TO RUN THE SCHEDULING ALGORITHM AND PLOTTING
        :param dataset:
        :param model:
        :return:
        """
        pass