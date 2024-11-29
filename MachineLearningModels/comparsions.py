import pandas as pd
import matplotlib.pyplot as plt
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from BatterSplit import batterSplit
from NormalLineScratch import LineatRegressionScratch
from linearRegressionlib import LinearRegression_lib
from NurealNetwork import N_network
from KNN import KNN_lib
from SVR import SVR_lib
from DecisionTree import DecisionTree_lib
from RandomForest import RandomForest_lib

# Dictionary to store MAE results
mae_results = {}

def save_mae_results(model_name, train_mae, test_mae):
    """
    Save the MAE results for each model.
    """
    mae_results[model_name] = {"train": train_mae, "test": test_mae}

def plot_comparison(mae_results): #TODO find the problem with decision tree
    """
    Plot training and testing MAE for all models.
    """
    models = list(mae_results.keys())
    train_mae = [mae_results[model]['train'] for model in models]
    test_mae = [mae_results[model]['test'] for model in models]

    # Plot Training MAE
    plt.figure(figsize=(10, 6))
    plt.bar(models, train_mae, color='skyblue')
    plt.title("Training MAE Comparison")
    plt.xlabel("Models")
    plt.ylabel("MAE")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot Testing MAE
    plt.figure(figsize=(10, 6))
    plt.bar(models, test_mae, color='salmon')
    plt.title("Testing MAE Comparison")
    plt.xlabel("Models")
    plt.ylabel("MAE")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    #data_visualization(df, "RunTime ")
    X_train, X_test, y_train, y_test = batterSplit(df, 'RunTime ', 0.2)
    GREEN = '\033[32m'
    RESET = '\033[0m'

    print(f"{GREEN}############### Normal Line from scratch ################{RESET}")
    model_linearRegressionScratch = LineatRegressionScratch()
    train_mae, test_mae = model_linearRegressionScratch.train(X_train, y_train, X_test, y_test)
    save_mae_results("NormalLine (Scratch)", train_mae, test_mae)

    print(f"{GREEN}############### Normal Line from library ################{RESET}")
    model_linearRegression_lib = LinearRegression_lib()
    train_mae, test_mae = model_linearRegression_lib.train(X_train, y_train, X_test, y_test)
    save_mae_results("NormalLine (Library)", train_mae, test_mae)

    print(f"{GREEN}############### KNN from library ########################{RESET}")
    model_knn_lib = KNN_lib()
    train_mae, test_mae = model_knn_lib.train(X_train, y_train, X_test, y_test)
    save_mae_results("KNN (Library)", train_mae, test_mae)

    print(f"{GREEN}############### SVR from library ########################{RESET}")
    model_svr_lib = SVR_lib()
    train_mae, test_mae = model_svr_lib.train(X_train, y_train, X_test, y_test)
    save_mae_results("SVR (Library)", train_mae, test_mae)

    print(f"{GREEN}############### Decision Tree from library ##############{RESET}")
    model_decisionTree_lib = DecisionTree_lib()
    train_mae, test_mae = model_decisionTree_lib.train(X_train, y_train, X_test, y_test)
    feature_importances, feature_names = model_decisionTree_lib.feature_importances, model_decisionTree_lib.feature_names
    save_mae_results("DecisionTree (Library)", train_mae, test_mae)

    print(f"{GREEN}############### Random Forest from library ##############{RESET}")
    model_randomForest_lib = RandomForest_lib()
    train_mae, test_mae = model_randomForest_lib.train(X_train, y_train, X_test, y_test)
    save_mae_results("RandomForest (Library)", train_mae, test_mae)

    print(f"{GREEN}############### Neural Network from library #############{RESET}")
    model_neuralNetwork = N_network()
    train_mae, test_mae = model_neuralNetwork.train(X_train, y_train, X_test, y_test)
    save_mae_results("NeuralNetwork (Library)", train_mae, test_mae)

    # plot important features
    model_decisionTree_lib.plot_feature_importances()
    # plot number of instances and features against the target
    data_visualization(df, "RunTime ")
    # Plot MAE comparison
    plot_comparison(mae_results)