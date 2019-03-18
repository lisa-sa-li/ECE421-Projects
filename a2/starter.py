import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(x):
    return np.maximum(0,x)

def GradReLU(x):
    return np.where(x <= 0, 0, 1)

def softmax(x):
    num = np.exp(x)
    denom = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.divide(num, denom)


def computeLayer(X, W, b):
    return np.dot(X, W) + np.repeat(np.transpose(b), X.shape[0], axis=0)

def CE(target, prediction):
    return -np.sum(target * np.log(prediction))/ prediction.shape[0]

def gradCE(target, prediction):
    return -np.sum(target/prediction)/ prediction.shape[0]

def initialize_params(size_i, size_h, size_o):

    W_h = np.random.normal(0, 2/(size_i+size_h), (size_i, size_h))
    b_h = np.zeros(shape=(size_h, 1))
    W_o = np.random.normal(0, 2/(size_h+size_o), (size_h, size_o))
    b_o = np.zeros(shape=(size_o, 1))

    wb_dict = {"W_h": W_h, "b_h": b_h, "W_o": W_o, "b_o": b_o}

    return wb_dict


# forward_propagation

def forward_propagation(data, params):
    # input data of size
    # params: dictionary containing weights and biases

    W_h = params['W_h']
    b_h = params['b_h']
    W_o = params['W_o']
    b_o = params['b_o']

    # fprop to calculate output
    Z_h = computeLayer(data, W_h, b_h)
    S_h = relu(Z_h)
    Z_o = computeLayer(S_h, W_o, b_o)
    S_o = softmax(Z_o)

    intermediates_dict = {"Z_h": Z_h, "S_h": S_h, "Z_o": Z_o, "S_o": S_o}

    return S_o, intermediates_dict


def backward_propagation(params, intermediates_dict, data, labels):
    N = data.shape[0]

    W_h = params['W_h']
    W_o = params['W_o']
    S_h = intermediates_dict['S_h']
    S_o = intermediates_dict['S_o']
    Z_h = intermediates_dict['Z_h']

    # Backward propagation: calculate dW_h, db_h, dW_o, db_o.
    dW_o = (1 / N) * np.matmul(S_h.T, S_o - labels) #shape: (Kx10)
    db_o = (1 / N) * np.sum(S_o - labels, axis=0)    #shape: (1x10)

    mat1 = np.multiply(GradReLU(Z_h), np.matmul(S_o - labels, W_o.T))
    dW_h = (1 / N) * np.matmul(data.T, mat1)

    db_h = (1 / N) * np.sum(mat1, axis=0)

    grads = {"dW_o": dW_o, "db_o": db_o, "dW_h": dW_h, "db_h": db_h}

    return grads


def update_params(params, grads, learning_rate=1, momentum_gamma=0.9):

    W_h = params['W_h']
    b_h = params['b_h']
    W_o = params['W_o']
    b_o = params['b_o']

    dW_o = grads['dW_o']
    db_o = grads['db_o']
    dW_h = grads['dW_h']
    db_h = grads['db_h']

    # Update rule for each parameter

    W_h = W_h - (momentum_gamma * W_h + learning_rate * dW_h)
    b_h = b_h - (momentum_gamma * b_h + learning_rate * db_h)
    W_o = W_o - (momentum_gamma * W_o + learning_rate * dW_o)
    b_o = b_o - (momentum_gamma * b_o + learning_rate * db_o)

    updated_params = {"W_h": W_h, "b_h": b_h, "W_o": W_o, "b_o": b_o}

    return params


def training_loop(data, labels, size_h, epochs):

    np.random.seed(9)

    params = initialize_params(data.shape[1], size_h, labels.shape[1])

    #Gradient descent
    for i in range(epochs):

        S_o, intermediates_dict = forward_propagation(data, params)

        cost = CE(labels, S_o)

        grads = backward_propagation(params, intermediates_dict, data, labels)

        params = update_params(params, grads)

        print("Loss after iteration %i: %f" % (i, cost))

def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    training_loop(trainData.reshape(trainData.shape[0], trainData.shape[1]*trainData.shape[2]), trainTarget, size_h=1000, epochs=10000)

if __name__ == "__main__":
    main()