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
from PlottingMethods.SchedulingPlotting import plot_turn_around_vs_waiting_time
import time
from colorama import Fore, Style, init
if __name__ == '__main__':
    init()  # initialize colorama
    data1_models = {}   # dictionary to store the models for dataset 1, the key is the model name and the value is the model object
    data2_models = {}   # dictionary to store the models for dataset 2, the key is the model name and the value is the model object
    data_models = [data1_models, data2_models]  # list of dictionaries to store the models for both datasets, this was created to reduce the code redundancy
    """
    if for example we want to access the linear regression model for dataset 1 we can do the following: data_models[0]['Linear Regression from scratch']
    basically calling the list that has the dictonary for dataset 1 and 2, than inside the dictionary we call the model using it is key
    """
    predictions_data1 = {}  # dictionary to store the predictions for dataset 1, the key is the model name and the value is the predicted dataframe
    predictions_data2 = {}  # dictionary to store the predictions for dataset 2, the key is the model name and the value is the predicted dataframe
    predictions = [predictions_data1, predictions_data2]    # list of dictionaries to store the predictions for both datasets, this was created to reduce the code redundancy
    GREEN = Fore.GREEN  # Green color
    RESET = Style.RESET_ALL  # Reset to default color
    RED = Fore.RED  # Red color
    YELLOW = Fore.YELLOW  # Yellow color
    CYAN = Fore.CYAN  # Cyan color
    MAGENTA = Fore.MAGENTA  # Magenta color

    # a function to train the model based on the user input based on the dataset the user choose
    def train_model(dataset, model=0, datasetint=0):
        """
        :param dataset: dataset dataframe to work on
        :param model: machine learning model including
        linear regression scratch = 1
        linear regression library = 2
        decision tree = 3
        random forest = 4
        KNN = 5
        SVR = 6
        Neural Network = 7
        0 for all models
        :param datasetint: integer that tell the function which dataset to work train and predict on
        :return: return nothing, just add the model and prediciton to the data_models and predictions dictionaries
        ths function was made to reduce the code redundancy
        """

        if model == 0:
            print(f"{RED}Training all models will take time please wait...{RESET}")
            train_model(dataset, 1, datasetint)
            train_model(dataset, 2, datasetint)
            train_model(dataset, 3, datasetint)
            train_model(dataset, 4, datasetint)
            train_model(dataset, 5, datasetint)
            train_model(dataset, 6, datasetint)
            train_model(dataset, 7, datasetint)
        if model == 1:
            print(f"{GREEN}######Training Linear Regression from scratch######{RESET}")
            if 'Linear Regression from scratch' in data_models[datasetint]: # check if the model has been trained before
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['Linear Regression from scratch'].mae_test}{RESET}") # print the MAE of the model

            else:
                # create the model object and train it and save it in the data_models dictionary
                linearRegScratch = LineatRegressionScratch()
                linearRegScratch.train(*dataset)
                data_models[datasetint]['Linear Regression from scratch'] = linearRegScratch
                # predict the data and save it in the predictions dictionary
                predictions[datasetint]['Linear Regression from scratch'] = linearRegScratch.predict(dataset[2])

        if model == 2:
            print(f"{GREEN}######Training Linear Regression from library######{RESET}")
            if 'Linear Regression from library' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['Linear Regression from library'].mae_test}{RESET}")
            else:
                linearRegLib = LinearRegression_lib()
                linearRegLib.train(*dataset)
                data_models[datasetint]['Linear Regression from library'] = linearRegLib
                predictions[datasetint]['Linear Regression from library'] = linearRegLib.predict(dataset[2])

        if model == 3:
            print(f"{GREEN}######Training Decision Tree######{RESET}")
            if 'Decision Tree' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['Decision Tree'].mae_test}{RESET}")
            else:
                decisionTree = DecisionTree_lib()
                decisionTree.train(*dataset)
                data_models[datasetint]['Decision Tree'] = decisionTree
                predictions[datasetint]['Decision Tree'] = decisionTree.predict(dataset[2])
        if model == 4:
            print(f"{GREEN}######Training Random Forest######{RESET}")
            if 'Random Forest' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['Random Forest'].mae_test}{RESET}")
            else:
                randomForest = RandomForest_lib()
                randomForest.train(*dataset)
                data_models[datasetint]['Random Forest'] = randomForest
                predictions[datasetint]['Random Forest'] = randomForest.predict(dataset[2])
        if model == 5:
            print(f"{GREEN}######Training KNN######{RESET}")
            if 'KNN' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['KNN'].mae_test}{RESET}")
            else:
                knn = KNN_lib()
                knn.train(*dataset)
                data_models[datasetint]['KNN'] = knn
                predictions[datasetint]['KNN'] = knn.predict(dataset[2])
        if model == 6:
            print(f"{GREEN}######Training SVR######{RESET}")
            if 'SVR' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['SVR'].mae_test}{RESET}")
            else:
                svr = SVR_lib()
                svr.train(*dataset)
                data_models[datasetint]['SVR'] = svr
                predictions[datasetint]['SVR'] = svr.predict(dataset[2])
        if model == 7:
            print(f"{GREEN}######Training Neural Network######{RESET}")
            if 'Neural Network' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['Neural Network'].mae_test}{RESET}")
            else:
                nn = N_network()
                nn.train(*dataset)
                data_models[datasetint]['Neural Network'] = nn
                predictions[datasetint]['Neural Network'] = nn.predict(dataset[2])
        pass

    # Read the data
    df = pd.read_csv(os.path.join('MachineLearningModels','processes_datasets.csv'))
    #Re index so index = JobID or basically start from 1 instead of 0
    df.index = df.index + 1
    # Preprocess the data  and create data 1 and 2
    data1, mapping1 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
         ], samples=100)
    data2, mapping2 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
        ], samples=1000)
    print(f"{GREEN}Dataset 1 ready with {RED}{data1.shape[0]}{GREEN} rows{RESET}")
    print(f"{GREEN}Dataset 2 ready with {RED}{data2.shape[0]}{GREEN} rows{RESET}")
    # Split the data
    X_train1, X_test1, y_train1, y_test1 = batterSplit(data1, 'RunTime ', 0.2)
    X_train2, X_test2, y_train2, y_test2 = batterSplit(data2, 'RunTime ', 0.2)
    split1 = [X_train1, y_train1, X_test1, y_test1]
    split2 = [X_train2, y_train2, X_test2, y_test2]
    print(f"{GREEN}both Data splitted with ratio {RED}0.2{RESET}")
    # main program
    while True:
        # ask the user which dataset to work on first
        print(f"{CYAN}Which dataset would you like to work?{RESET}")
        print(f"{YELLOW}1{RESET}. {MAGENTA}Dataset 1 with 100 samples{RESET}")
        print(f"{YELLOW}2{RESET}. {MAGENTA}Dataset 2 with 1000 samples{RESET}")
        print(f"{YELLOW}3{RESET}. {RED}Exit{RESET}")
        choice = input() # get the user input to choose the dataset
        # this string will be printed at multiple places in the code so it is better to store it in a variable to reduce redundancy
        models_print = (f"{YELLOW}0{RESET}. {MAGENTA}All{RESET}\n"
                        f"{YELLOW}1{RESET}. {MAGENTA}Linear Regression from scratch{RESET}\n"
                        f"{YELLOW}2{RESET}. {MAGENTA}Linear Regression from library{RESET}\n"
                        f"{YELLOW}3{RESET}. {MAGENTA}Decision Tree{RESET}\n"
                        f"{YELLOW}4{RESET}. {MAGENTA}Random Forest{RESET}\n"
                        f"{YELLOW}5{RESET}. {MAGENTA}KNN{RESET}\n"
                        f"{YELLOW}6{RESET}. {MAGENTA}SVR{RESET}\n"
                        f"{YELLOW}7{RESET}. {MAGENTA}Neural Network{RESET}\n"
                        f"{YELLOW}8{RESET}. {RED}Exit{RESET}")

        if choice == '1':
            print(f"{GREEN}######Working on Dataset 1######{RESET}")
            while True: # loop to train the model based on the user input
                print(f"{CYAN}Which model would you like to train?{RESET}")
                print(models_print)
                model_choice = input() # get the user input to choose the model
                try: # make sure the user input is valid
                    int(model_choice)
                except ValueError: # if the user input is not a number
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                if int(model_choice) == 8: # if the user choose to exit
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(0.5)
                    break
                if int(model_choice) not in range(9): # if the user input is not in the legal range
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                # train the model based on the user input after passing all the checks
                train_model(split1, int(model_choice), 0)
                time.sleep(0.5)
            # make sure the user trained at least one model before scheduling
            if len(predictions_data1) == 0:
                print(f"{RED}No model has been trained, will train default model{RESET}")
                train_model(split1, 1, 0)
                time.sleep(0.5)
            # schedule the models, basically adding a new column to that store the waiting time and turn around time for each scheduling algorithm

            print(f"{RED}scheduling...{RESET}")
            for key, value in predictions_data1.items():
                try: # make sure the data is not already scheduled
                    value = denormlize_data(value, mapping1) # denormlize the data
                    temp = [[],[]] # create a temporary list to store the data
                    df, avg_waiting_time, avg_turn_around_time = SJF(value, value.columns[-1]) # schedule the data using SJF
                    temp[0] = [df, avg_waiting_time, avg_turn_around_time] # store the scheduled data and the average waiting time and turn around time using SJF at the first place in the list
                    df, avg_waiting_time, avg_turn_around_time = FCFS(value, value.columns[-1]) # schedule the data using FCFS
                    temp[1] = [df, avg_waiting_time, avg_turn_around_time] # store the scheduled data and the average waiting time and turn around time using FCFS at the second place in the list
                    predictions_data1[key] = temp # store the list in the predictions_data1 dictionary, this will turn the value which used to be just a dataframe to a list
                except:
                    print(f"{RED}{key} already scheduled skipping to next...{RESET}")
                    continue
            #loop thro the dictionary predictions_data1 and
            time.sleep(0.5)
            while True:
                print(f"{CYAN}Which model would you like to plot the schedule for?{RESET}")
                print(f"{YELLOW}0{RESET}. {MAGENTA}All available models{RESET}")
                value = list(predictions_data1.values()) # get the values of the dictionary and store it in a list
                keys = list(predictions_data1.keys()) # get the keys of the dictionary and store it in a list
                for index, key in enumerate(keys, start=1): # loop thro the keys and print them (this make sure to catch what models were trained and have dynamic input)
                    print(f"{YELLOW}{index}{RESET}. {MAGENTA}{key}{RESET}")
                print(f"{YELLOW}{(len(value)+1)}{RESET}. {RED}Exit{RESET}")

                choice = input()
                # make sure the user input is valid and in legal range
                try:
                    int(choice)
                except ValueError:
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                if int(choice) not in range(0,len(value)+2):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                if int(choice) == index + 1: # if the user choose to exit
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(0.5)
                    break
                if int(choice) == 0: # if the user choose to plot all the models which is not implemented yet TODO
                    print(f"{GREEN}Plotting all models not implemented yet :){RESET}")
                    time.sleep(0.5)
                    continue
                print(f"{GREEN}Plotting {keys[int(choice)-1]}...{RESET}")
                plot_turn_around_vs_waiting_time(value[int(choice)-1], keys[int(choice)-1])
                time.sleep(0.5)



        elif choice == '2':
            print(f"{GREEN}######Working on Dataset 2######{RESET}")
            while True:
                print(f"{CYAN}Which model would you like to train?{RESET}")
                print(models_print)
                model_choice = input()
                try:
                    int(model_choice)
                except ValueError:
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                if int(model_choice) == 8:
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(0.5)
                    break
                if int(model_choice) not in range(9):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                train_model(split2, int(model_choice), 1)
                time.sleep(0.5)
            # TODO tell the user what model has been trained and ask which one he wish to schedule
            if len(predictions_data2) == 0:
                print(f"{RED}No model has been trained, will train default model{RESET}")
                train_model(split2, 1, 1)
                time.sleep(0.5)
            print(f"{RED}scheduling...{RESET}")
            for key, value in predictions_data2.items():
                try:
                    value = denormlize_data(value, mapping2)
                    temp = [[],[]]
                    df, avg_waiting_time, avg_turn_around_time = SJF(value, value.columns[-1])
                    print(value.columns[-1])
                    temp[0] = [df, avg_waiting_time, avg_turn_around_time]
                    df, avg_waiting_time, avg_turn_around_time = FCFS(value, value.columns[-1])
                    temp[1] = [df, avg_waiting_time, avg_turn_around_time]
                    predictions_data2[key] = temp
                except:
                    print(f"{RED}{key} already scheduled skipping to next...{RESET}")
                    continue
            # loop thro the dictionary predictions_data1 and plot the schedule
            time.sleep(1)
            while True:
                print(f"{CYAN}Which model would you like to plot the schedule for?{RESET}")
                print(f"{YELLOW}0{RESET}. {MAGENTA}All available models{RESET}")
                value = list(predictions_data2.values())
                keys = list(predictions_data2.keys())
                for index, key in enumerate(keys, start=1):
                    print(f"{YELLOW}{index}{RESET}. {MAGENTA}{key}{RESET}")
                print(f"{YELLOW}{(len(value)+1)}{RESET}. {RED}Exit{RESET}")

                choice = input()
                # make sure the user input is valid and in legal range
                try:
                    int(choice)
                except ValueError:
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                if int(choice) not in range(0,len(value)+2):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(0.5)
                    continue
                if int(choice) == index + 1:
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(0.5)
                    break
                if int(choice) == 0:
                    print(f"{GREEN}Plotting all models not implemented yet :){RESET}")
                    time.sleep(0.5)
                    continue
                print(f"{GREEN}Plotting {keys[int(choice)-1]}...{RESET}")
                plot_turn_around_vs_waiting_time(value[int(choice)-1], keys[int(choice)-1])
                time.sleep(0.5)
        elif choice == '3':
            print(f"{RED}Exiting...{RESET}")
            time.sleep(1.5)
            break
        else:
            print(f"{RED}Invalid choice{RESET}")
            time.sleep(1.5)
            continue



