import numpy as np
import pandas as pd
from DataPreProcess import data_preprocess
from sklearn.model_selection import train_test_split
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.input_layer = None
        self.addlayer = -1
        self.weights = []
        self.biases = []
        self.activations = []
        self.Z = []
        self.A = []

    def input_layer_add(self, neurons):
        """
        Adds the input layer to the neural network.

        Parameters:
        neurons: int
            The number of neurons in the input layer.

        Returns:
        str
            Message confirming the input layer addition.
        """
        # make sure the input layer is not added before
        if self.addlayer == 0:
            raise ValueError("Input layer already added")
        # make sure the input layer has at least 1 neuron
        if neurons <= 0:
            raise ValueError("neurons must be greater than 0")
        # add the input layer
        self.input_layer = neurons
        self.addlayer = 0
        return "Input layer added"

    def add(self, layer):
        """
        Adds a layer to the neural network.

        Parameters:
        layer: tuple
            A tuple with the number of neurons and the activation function.

        Returns:
        str
            Message confirming the layer addition.
        """
        # make sure the layer is tuple and has 2 elements 1 is number 2nd is a string (activation function) either "Linear" or "relu"
        if not isinstance(layer, tuple) or not isinstance(layer[0], int) or not isinstance(layer[1], str):
            raise ValueError("Layer must be a tuple of 2 elements (int, str) current activation function supported are Linear and relu and input for the input layer")
        # make sure the first layer is input layer and no other layer is input layer
        neurons, activation = layer
        if activation.lower() not in ["linear", "relu"]:
            raise ValueError("activation function must be either Linear or relu")
        if self.addlayer == -1:
            raise ValueError("You must add input layer first")
        # make sure the number of neurons is greater than 0
        if neurons <= 0:
            raise ValueError("neurons must be greater than 0")
        # add the layer
        if len(self.layers) == 0:
            previous_neurons = self.input_layer
        else:
            previous_neurons = self.layers[-1]
        self.layers.append(neurons)
        self.activations.append(activation)
        self.weights.append(np.random.uniform(-0.5, 0.5, (neurons, previous_neurons)))
        self.biases.append(np.random.uniform(-0.5, 0.5, (neurons, 1)))
        return f"Hidden layer number {len(self.layers)} added with {neurons} neurons and {activation} activation weights shape: {self.weights[-1].shape} biases shape: {self.biases[-1].shape}"

    def linear(self, x):
        """
        Linear activation function.
        """
        return x

    def relu(self, x):
        """
        ReLU activation function.
        """
        return np.maximum(0, x)

    def mean_absolute_error(self, y_true, y_pred):
        """
        Mean absolute error loss function.

        Parameters:
        y_true: numpy array
            The true target values.
        y_pred: numpy array
            The predicted target values.

        Returns:
        float
            The mean absolute error.
        """
        return np.mean(np.abs(y_true - y_pred))

    def mean_squared_error(self, y_true, y_pred):
        """
        Mean squared error loss function.

        Parameters:
        y_true: numpy array
            The true target values.
        y_pred: numpy array
            The predicted target values.

        Returns:
        float
            The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def relu_derivative(self, x):
        """
        Derivative of the ReLU activation function.

        Parameters:
        x: numpy array
            The input data.

        Returns:
        numpy array
            The derivative of the ReLU activation function.
        """
        return x > 0

    def mse_derivative(self, y_true, y_pred):
        """
        Derivative of the mean squared error loss function.

        Parameters:
        y_true: numpy array
            The true target values.
        y_pred: numpy array
            The predicted target values.
        n: number of samples

        Returns:
        numpy array
            The derivative of the mean squared error loss function.
        """
        n = y_true.shape[0]
        return (2/n) * (y_pred - y_true)

    def mae_derivative(self, y_true, y_pred):
        """
        Derivative of the mean absolute error loss function.

        Parameters:
        y_true: numpy array
            The true target values.
        y_pred: numpy array
            The predicted target values.
        n: number of samples

        Returns:
        numpy array
            The derivative of the mean absolute error loss function.
        """
        n = y_true.shape[0]
        return (1/n) * np.sign(y_pred - y_true)

    def forward_propagation(self, X):
        """
        Forward propagation of the neural network.

        Parameters:
        X: numpy array
            The input data.

        Returns:
        numpy array
            The output of the neural network.
        """
        self.Z = [] # Store the Z (not activated) values for backpropagation
        self.A = [] # Store the A (activated) values for backpropagation
        for i in range(0, len(self.layers)):
            if i == 0:
                W = self.weights[i]
                b = self.biases[i]
                activation = self.activations[i]
                Z = W.dot(X.T) + b
                self.Z.append(Z)
                if activation.lower() == "linear":
                    self.A.append(self.linear(Z))
                elif activation.lower() == "relu":
                    self.A.append(self.relu(Z))
            else:
                W = self.weights[i]
                b = self.biases[i]
                activation = self.activations[i]
                #print(f"Layer {i+1} W shape: {W.shape}, b shape: {b.shape} activation: {activation} previous layer: {self.A[i-1].shape}")
                Z = W.dot(self.A[i-1]) + b
                #Store the Z values for backpropagation in the corresponding layer
                self.Z.append(Z)
                #print(f"Layer {i+1} Z shape: {Z.shape}")
                if activation.lower() == "linear":
                    self.A.append(self.linear(Z))
                elif activation.lower() == "relu":
                    self.A.append(self.relu(Z))
        return self.A[-1]

    def backward_propagation(self, X, y, learning_rate, loss_function):
        """
        Backward propagation of the neural network.
        :param X:
        :param y:
        :param loss_function:
        :param learning_rate:
        :return: updated weights and biases
        """

        for i in range(len(self.layers) - 1, -1, -1):
            #print(f"layer {i + 1} shape: {self.weights[i].shape}")
            # handle output layer
            if i == len(self.layers) - 1:
                # compute the derivative of the loss function
                if loss_function.lower() == "mae":
                    dZ = self.mae_derivative(y, self.A[-1])
                elif loss_function.lower() == "mse":
                    dZ = self.mse_derivative(y, self.A[-1])
                """
                # compute the derivative of the activation function
                if self.activations[i].lower() == "linear":
                    dZ = dZ
                elif self.activations[i].lower() == "relu":
                    dZ = dZ * self.relu_derivative(self.Z[i])
                """
                dw = (1 / y.shape[0]) * dZ.dot(self.A[i-1].T)
                db = (1 / y.shape[0]) * np.sum(dZ, axis=1, keepdims=True)
                self.weights[i] = self.weights[i] - learning_rate * dw
                self.biases[i] = self.biases[i] - learning_rate * db
            # handle hidden layer after the input layer
            elif i == 0:
                if self.activations[i].lower() == "linear":
                    dZ = self.weights[i + 1].T.dot(dZ)
                elif self.activations[i].lower() == "relu":
                    dZ = self.weights[i + 1].T.dot(dZ) * self.relu_derivative(self.Z[i])
                dw = (1 / y.shape[0]) * dZ.dot(X)
                db = (1 / y.shape[0]) * np.sum(dZ, axis=1, keepdims=True)
                self.weights[i] = self.weights[i] - learning_rate * dw
                self.biases[i] = self.biases[i] - learning_rate * db
            # handle hidden layers
            else:
                if self.activations[i].lower() == "linear":
                    dZ = self.weights[i + 1].T.dot(dZ)
                elif self.activations[i].lower() == "relu":
                    dZ = self.weights[i + 1].T.dot(dZ) * self.relu_derivative(self.Z[i])
                dw = (1 / y.shape[0]) * dZ.dot(self.A[i-1].T)
                db = (1 / y.shape[0]) * np.sum(dZ, axis=1, keepdims=True)
                self.weights[i] = self.weights[i] - learning_rate * dw
                self.biases[i] = self.biases[i] - learning_rate * db
        #print(f"Layer {i + 1} dw shape: {dw.shape}, db shape: {db.shape}, dz shape: {dZ.shape}")







    def train(self, X, y, epochs, learning_rate, loss_function):
        """
        Trains the neural network.

        Parameters:
        X: numpy array
            The input data.
        y: numpy array
            The target data.
        epochs: int
            The number of epochs to train the neural network.
        learning_rate: float
            The learning rate of the neural network.
        """

        # make sure the inputs are legal
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if len(self.layers) < 2:
            raise ValueError("You must add at least Two layer to the neural network")
        if loss_function.lower() not in ["mae", "mse"]:
            raise ValueError("loss function must be mae or mse for now")
        # train the neural network
        self.forward_propagation(X)
        for epoch in range(epochs):
            self.backward_propagation(X, y, learning_rate, loss_function)
            self.forward_propagation(X)
            loss = self.mean_absolute_error(y, self.A[-1])
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")

if __name__ == '__main__':
    nn = NeuralNetwork()
    # seed for reproducibility
    np.random.seed(42)
    df = pd.read_csv('processes_datasets.csv')
    df, mapping = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=100)
    X = df.drop('RunTime ', axis=1)
    y = df['RunTime ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X.shape[1])
    print(nn.input_layer_add(X.shape[1]))
    print(nn.add((64, "linear")))
    print(nn.add((32, "linear")))
    print(nn.add((16, "linear")))
    print(nn.add((8, "linear")))
    print(nn.add((1, "linear")))
    print(nn.layers)
    print(nn.activations)
    print(X_train.to_numpy().shape)
    #print(nn.forward_propagation(X_train.to_numpy()))
    print(y_train.to_numpy().shape[0])
    print(nn.train(X_train.to_numpy(), y_train.to_numpy(), 100, 0.1, "mae"))

# TODO NEVER DO THIS AGAIN