import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import os
from plotting import plot
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
    e_x = np.exp(x)
    denom = e_x.sum(axis=1)
    denom = np.tile(denom, (10, 1)).T
    return np.divide(e_x, denom)

def computeLayer(X, W, b):
    return np.matmul(X, W) + np.tile(b, (X.shape[0], 1))

def CE(target, prediction):
    N = target.shape[0]
    return -(1/N) *np.sum(np.multiply(target, np.log(prediction)))

def gradCE(target, prediction):
    N = target.shape[0]
    return (-1 / N) * np.sum(np.division(target, prediction))


# load the data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()


# Convert labels to one-hot
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)


# reshape the input data

trainData = np.reshape(trainData, (trainData.shape[0], -1))
validData = np.reshape(validData, (validData.shape[0], -1))
testData = np.reshape(testData, (testData.shape[0], -1))

# Hyperparameters

epochs = 50
hidden_size = 1000
lr = 0.01
gamma = 0.9

# Initialize weight and bias matrices

W_o = np.random.normal(0, np.sqrt(2/(hidden_size + 10)), (hidden_size, 10))

b_o = np.zeros((1, 10))

W_h = np.random.normal(0, np.sqrt(2/(784 + hidden_size)), (784, hidden_size))

b_h = np.zeros((1, hidden_size))

# Initialize the momentum matrices

v_Wo = np.full((W_o.shape), 1e-5)

v_Wh = np.full((W_h.shape), 1e-5)

v_bo = np.full((b_o.shape), 1e-5)

v_bh = np.full((b_h.shape), 1e-5)

# perform forward pass

def forwardPass(data, W_o, b_o, W_h, b_h):

    '''

    :param data: N x 784
    :param W_o: K x 10
    :param b_o: 1 x 10
    :param W_h: 784 x K
    :param b_h: 1 x K
    :return: into_hidden, outof_hidden, into_output, outof_output
    '''

    into_hidden = computeLayer(data, W_h, b_h)    # N x K

    outof_hidden = relu(into_hidden)            # N x K

    into_output = computeLayer(outof_hidden, W_o, b_o)      # N x 10

    outof_output = softmax(into_output)         # N x 10

    return into_hidden, outof_hidden, into_output, outof_output



def backprop_gradients(data, labels, S_h, X_h, X_o, W_o):

    N = labels.shape[0]

    dW_o = (1 / N) * np.matmul(X_h.T, X_o - labels)  # shape: (Kx10)
    db_o = (1 / N) * np.sum(X_o - labels, axis=0)  # shape: (1x10)

    mat1 = np.multiply(GradReLU(S_h), np.matmul(X_o - labels, W_o.T))
    dW_h = (1 / N) * np.matmul(data.T, mat1)

    db_h = (1 / N) * np.sum(mat1, axis=0)

    return dW_o, db_o, dW_h, db_h


def update_param(v, W, gamma, alpha, grad):

    vnew = gamma * v + alpha * grad

    W = W - vnew

    return W, vnew

def calculateLoss_Accuracy(predictions, labels):
    '''

    :param predictions: matrix of predictions from output of softmax
    :param labels: matrix of one-hot labels
    :return: accuracy and loss for this epoch
    '''

    pred_array = np.argmax(predictions, axis=1)
    labels_array = np.argmax(labels, axis=1)

    accuracy = (pred_array == labels_array).sum() / labels.shape[0]

    loss = CE(labels, predictions)

    return loss, accuracy



train_loss_list = []
val_loss_list = []
test_loss_list = []
train_acc_list = []
val_acc_list = []
test_acc_list = []

for epoch in range(epochs):

    # do a forward pass

    S_h, X_h, S_o, X_o = forwardPass(trainData, W_o, b_o, W_h, b_h)

    dW_o, db_o, dW_h, db_h = backprop_gradients(trainData, trainTarget, S_h, X_h, X_o, W_o)

    # update weights and biases

    W_o, v_Wo = update_param(v_Wo, W_o, gamma, lr, dW_o)
    b_o, v_bo = update_param(v_bo, b_o, gamma, lr, db_o)

    W_h, v_Wh = update_param(v_Wh, W_h, gamma, lr, dW_h)
    b_h, v_bh = update_param(v_bh, b_h, gamma, lr, db_h)

    # calculate losses and accuracies

    train_loss, train_acc  = calculateLoss_Accuracy(forwardPass(trainData, W_o, b_o, W_h, b_h)[3], trainTarget)
    valid_loss, valid_acc = calculateLoss_Accuracy(forwardPass(validData, W_o, b_o, W_h, b_h)[3], validTarget)
    test_loss, test_acc = calculateLoss_Accuracy(forwardPass(testData, W_o, b_o, W_h, b_h)[3], testTarget)


    print("Epoch: {}, | Training loss: {:.5f} | Validation loss: {:.5f} | Test loss: {:.5f}  "
          "Training Accuracy: {:.5f} | Validation Accuracy: {:.5f} | Test Accuracy: {:.5f} "
          .format(epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

    train_loss_list.append(train_loss)
    val_loss_list.append(valid_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(valid_acc)
    test_acc_list.append(test_acc)



plot(epochs, train_loss_list, val_loss_list, test_loss_list, train_acc_list, val_acc_list, test_acc_list)





















