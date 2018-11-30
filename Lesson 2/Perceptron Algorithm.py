import pandas as pd
import numpy as np

np.random.seed(42)

def stepFunction(t):
    """
    Step function : a function that increases or decreases abruptly from one constant value to another.
    Return 1 if greater than 0, else return 0
    :param t: prediction
    :return: Return 1 if prediction is greater than 0, else return 0
    """
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    """
    Prediction function which returns 1 or 0
    :param X: Input Features
    :param W: Weights
    :param b: bias
    :return: predicted value of y
    """
    return stepFunction((np.matmul(X, W) + b)[0])

def perceptronStep(X, y, W, b, learn_rate=0.01):
    """
    Update the weights and bias W, b, according to the perceptron algorithm,
    :param X: data
    :param y: labels
    :param W: weights
    :param b: bias
    :param learn_rate: learning rate
    :return: Updated weights and bias
    """
    for i in range(len(X)):
        a = y[i]
        y_hat = prediction(X[i], W, b)
        if y[i] - y_hat == 1:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate
    return W, b


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    """
    This function runs the perceptron algorithm repeatedly on the dataset,
    and returns a few of the boundary lines obtained in the iterations,
    for plotting purposes.
    :param X: data
    :param y: labels
    :param learn_rate: learning rate
    :param num_epochs: number of epochs to be run
    :return: boundary lines obtained in the iterations.
    """
    x_min, x_max = min(X[0]), max(X[0])
    y_min, y_max = min(X[1]), max(X[1])

    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max

    X = X.values
    y = y.values

    boundary_lines = []
    for i in range(num_epochs):

        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))

    return boundary_lines

if __name__ == '__main__':
    file_path = '/home/amog/Code/Pytorch_FB_Udacity_Challenge/data/data_perceptron_algorithm.csv'
    data = pd.read_csv(file_path,header=None)
    X = data.iloc[:,0:2]
    y = data.iloc[:,2:]
    boundary_lines = trainPerceptronAlgorithm(X,y)