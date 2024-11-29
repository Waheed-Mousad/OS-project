import pandas as pd
from DataPreProcess import data_preprocess, data_visualization, denormlize_data
from BatterSplit import batterSplit
import os
import numpy as np
import random
import warnings
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
# Suppress specific warnings
warnings.filterwarnings('ignore', message=".*tf.reset_default_graph is deprecated.*")  # Suppress tf.reset_default_graph deprecation warning sadly not working
warnings.filterwarnings('ignore', message=".*When using Sequential models, prefer using an.*")  # Suppress Sequential model warning
class N_network:
    """
    Neural network model from Tensorflow, have train and predict methods
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
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['PYTHONHASHSEED'] = str(42)
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        tf.keras.backend.clear_session()
        self.target_col = y_train.columns[0]
        self.feature_names = X_train.columns
        # Create a neural network
        num_features = X_train.shape[1]
        NN = Sequential(
            [Dense(num_features*8, input_shape=(num_features,)),  # First Hidden layer, I choose feature number times 8 for layer size
             Dense(num_features*4, ),  # Second Hidden layer, I choose half the previous layer size for this one
             Dense(num_features*2, ),  # Third Hidden layer, I choose half the previous layer size for this one
             Dense(num_features, ),  # fpurth layer, also half the previous
             Dense(1)])  # Output Layer, no activation for regression
        # Linear activation on the hidden layer were tested and the results were the best (for this specific use case)
        # Compile the model
        NN.compile(optimizer='adam', loss='mae', metrics=['mae'])
        # Early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=100,
                                       restore_best_weights=True)  # early stopping criteria
        # Fit the model
        NN.fit(X_train, y_train, epochs=1000, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping],
               shuffle=False, verbose=False)
        # Predict the data
        start = time.time()
        loss, mae_test = NN.evaluate(X_test, y_test, verbose=False)
        end = time.time()
        loss, mae_train = NN.evaluate(X_train, y_train, verbose=False)
        YELLOW = "\033[33m"  # Yellow color
        RED = "\033[31m"  # Red color
        RESET = "\033[0m"  # Reset to default color
        print(
            f"{YELLOW}(Neural Network from Tensorflow){RESET} {RED}MAE{RESET} for training data: {RED}{mae_train}{RESET}")
        print(
            f"{YELLOW}(Neural Network from Tensorflow){RESET} {RED}MAE{RESET} for testing data: {RED}{mae_test}{RESET}")
        print(f"{YELLOW}Time taken to predict the testing data:{RESET} {RED}{end - start} seconds{RESET}")
        self.model = NN
        self.mae_train = mae_train
        self.mae_test = mae_test

    def predict(self, X):
        y_pred = self.model.predict(X, verbose=False)
        X[self.target_col] = y_pred
        return X






if __name__ == '__main__':
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df,target_col=[
    'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
    'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    X_train, X_test, y_train, y_test = batterSplit(df, 'RunTime ', 0.2)
    model = N_network()
    model.train(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    print(y_pred)
