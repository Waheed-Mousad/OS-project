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

if __name__ == '__main__':
    data1_models = {}
    data2_models = {}
    data_models = [data1_models, data2_models]
    predictions_data1 = {}
    predictions_data2 = {}
    predictions = [predictions_data1, predictions_data2]
    GREEN = "\033[32m"  # Green color
    RESET = "\033[0m"  # Reset to default color
    RED = "\033[31m"  # Red color
    YELLOW = "\033[33m"  # Yellow color
    CYAN = "\033[36m"  # Cyan color
    MAGENTA = "\033[35m"  # Magenta color

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
        :return:
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
            if 'Linear Regression from scratch' in data_models[datasetint]:
                print(f"{RED}Model already trained{RESET}")
                print(f"{GREEN}Model test MAE:{RED} {data_models[datasetint]['Linear Regression from scratch'].mae_test}{RESET}")

            else:
                linearRegScratch = LineatRegressionScratch()
                linearRegScratch.train(*dataset)
                data_models[datasetint]['Linear Regression from scratch'] = linearRegScratch
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
    #Re index so index = JobID
    df.index = df.index + 1
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
        choice = input()
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
            while True:
                print(f"{CYAN}Which model would you like to train?{RESET}")
                print(models_print)
                model_choice = input()
                try:
                    int(model_choice)
                except ValueError:
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(1.5)
                    continue
                if int(model_choice) == 8:
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(1.5)
                    break
                if int(model_choice) not in range(9):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(1.5)
                    continue
                train_model(split1, int(model_choice), 0)
                time.sleep(1)
            # TODO tell the user what model has been trained and ask which one he wish to schedule show schedule for
            print(f"{RED}scheduling...{RESET}")

            for key, value in predictions_data1.items():
                try:
                    value = denormlize_data(value, mapping1)
                    temp = [[],[]]
                    df, avg_waiting_time, avg_turn_around_time = SJF(value, value.columns[-1])
                    temp[0] = [df, avg_waiting_time, avg_turn_around_time]
                    df, avg_waiting_time, avg_turn_around_time = FCFS(value, value.columns[-1])
                    temp[1] = [df, avg_waiting_time, avg_turn_around_time]
                    predictions_data1[key] = temp
                except:
                    print(f"{RED}{key} already scheduled skipping to next...{RESET}")
                    continue
            #loop thro the dictionary predictions_data1 and
            time.sleep(1)
            while True:
                print(f"{CYAN}Which model would you like to plot the schedule for?{RESET}")
                print(f"{YELLOW}0{RESET}. {MAGENTA}All available models{RESET}")
                value = list(predictions_data1.values())
                keys = list(predictions_data1.keys())
                for index, key in enumerate(keys, start=1):
                    print(f"{YELLOW}{index}{RESET}. {MAGENTA}{key}{RESET}")
                print(f"{YELLOW}{(len(value)+1)}{RESET}. {RED}Exit{RESET}")

                choice = input()
                # make sure the user input is valid and in legal range
                try:
                    int(choice)
                except ValueError:
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(1.5)
                    continue
                if int(choice) not in range(0,len(value)+2):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(1.5)
                    continue
                if int(choice) == index + 1:
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(1.5)
                    break
                if int(choice) == 0:
                    print(f"{GREEN}Plotting all models not implemented yet :){RESET}")
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
                    time.sleep(1.5)
                    continue
                if int(model_choice) == 8:
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(1.5)
                    break
                if int(model_choice) not in range(9):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(1.5)
                    continue
                train_model(split2, int(model_choice), 1)
                time.sleep(1)
            # TODO tell the user what model has been trained and ask which one he wish to schedule
            print(f"{RED}scheduling...{RESET}")
            for key, value in predictions_data2.items():
                try:
                    value = denormlize_data(value, mapping2)
                    temp = [[],[]]
                    df, avg_waiting_time, avg_turn_around_time = SJF(value, value.columns[-1])
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
                    time.sleep(1.5)
                    continue
                if int(choice) not in range(0,len(value)+2):
                    print(f"{RED}Invalid choice{RESET}")
                    time.sleep(1.5)
                    continue
                if int(choice) == index + 1:
                    print(f"{RED}Exiting...{RESET}")
                    time.sleep(1.5)
                    break
                if int(choice) == 0:
                    print(f"{GREEN}Plotting all models not implemented yet :){RESET}")
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



