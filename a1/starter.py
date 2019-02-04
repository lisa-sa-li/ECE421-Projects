import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from normal_equations import WLS
from plotting import plot

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
n_samples = trainData.shape[0]

validData = np.reshape(validData, (validData.shape[0], -1))

testData = np.reshape(testData, (testData.shape[0], -1))

#storing losses and accuracies

trainloss_list = []
valloss_list = []
testloss_list = []

train_acc_list = []
val_acc_list = []
test_acc_list = []


# parameters
W = np.zeros((28, 28))
b = 0
lrs = [0.005, 0.001, 0.0001]
error_tolerance = 1e-7
epochs = 5000
reg = [0, 0.001, 0.1, 0.5]



def MSE(W, b, x, y, reg):
    '''

    :param W: weight matrix
    :param b: bias matrix
    :param x: data matrix   N x (d+1). N data points, each one having dimension d+1
    :param y: labels (0,1)
    :param reg: regularization constant
    :return: total loss
    '''


    N = y.shape[0]

    total_loss = (1/(2*N))*(np.linalg.norm(np.matmul(x, W) + b - y))**2 + 0.5*reg*np.sum(np.square(W))



    return total_loss


def gradMSE(W, b, x, y, reg):
    '''

    :param W:
    :param b:
    :param x:
    :param y:
    :param reg:
    :return: gradient wrt W, gradient wrt b
    '''

    N = y.shape[0]

    f_w = (1/N)*np.matmul(np.transpose(x), (np.matmul(x, W) + b - y)) + reg*W

    f_b = (1/N)*np.sum((np.matmul(x, W) + b - y))

    return [f_w, f_b]

def crossEntropyLoss(W, b, x, y, reg):

    N = y.shape[0]

    loss1 = -np.mean(y * np.log(logistic_y_hat(W, x, b)) + (1-y)*np.log(1-logistic_y_hat(W, x, b)))
    loss2 = 0.5*reg*(np.linalg.norm(W))**2

    total_loss = loss1 + loss2

    return total_loss

def gradCE(W, b, x, y, reg):

    N = y.shape[0]

    dw = (1/N) * ((x.T) @ (logistic_y_hat(W, x, b) - y)) + reg*W

    db = (1/N)*np.sum(logistic_y_hat(W, x, b) - y)

    return [dw, db]


def sigmoid(z):
    return 1/(1+np.exp(-z))


def logistic_y_hat (W, x, b):
    return sigmoid(x@W + b)


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType = None):
    '''

    :param W: weight matrix
    :param b: bias vector
    :param trainingData: (3500, 28, 28)
    :param trainingLabels: (3500, 1)
    :param alpha:
    :param iterations: num epochs
    :param reg: regularization constant
    :param EPS: error tolerance
    :return: optimized weight and bias vectors
    '''

    start_time = time.time()

    #Initialize weight matrix with random
    trainingData = np.reshape(trainingData, (trainingData.shape[0], -1))

    W = np.reshape(W, (W.shape[0] * W.shape[1], -1))


    N = trainingLabels.shape[0]




    for epoch in range(iterations):
        #one pass through entire dataset

        if lossType == "MSE":
            gradients = gradMSE(W, b, trainingData, trainingLabels, reg)

            train_loss = MSE(W, b, trainingData, trainingLabels, reg)
            val_loss = MSE(W, b, validData, validTarget, reg)
            test_loss = MSE(W, b, testData, testTarget, reg)

        elif lossType == "CE":
            gradients = gradCE(W, b, trainingData, trainingLabels, reg)

            train_loss = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
            val_loss = crossEntropyLoss(W, b, validData, validTarget, reg)
            test_loss = crossEntropyLoss(W, b, testData, testTarget, reg)


        grad_weights = gradients[0]
        grad_biases = gradients[1]

        W = W - alpha*grad_weights #(784x1)
        b = b - alpha*grad_biases #(784x1)


        if lossType == "MSE":

            predicted_train = np.matmul(trainingData, W) + b
            predicted_val = np.matmul(validData, W) + b
            predicted_test = np.matmul(testData, W) + b

            predicted_train[predicted_train > 0] = 1
            predicted_train[predicted_train < 0] = 0

            predicted_val[predicted_val > 0] = 1
            predicted_val[predicted_val < 0] = 0

            predicted_test[predicted_test > 0] = 1
            predicted_test[predicted_test < 0] = 0

        elif lossType == "CE":

            predicted_train = np.expand_dims(np.array([1 if i > 0.5 else 0 for i in logistic_y_hat(W, trainingData, b)]), axis=1)
            predicted_val = np.expand_dims(np.array([1 if i > 0.5 else 0 for i in logistic_y_hat(W, validData, b)]), axis=1)
            predicted_test = np.expand_dims(np.array([1 if i > 0.5 else 0 for i in logistic_y_hat(W, testData, b)]), axis=1)


        train_acc = np.sum(predicted_train == trainingLabels) / N
        val_acc = np.sum(predicted_val == validTarget) / validTarget.shape[0]
        test_acc = np.sum(predicted_test == testTarget) / testTarget.shape[0]

        #storing losses and accuracies
        trainloss_list.append(train_loss)
        valloss_list.append(val_loss)
        testloss_list.append(test_loss)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)


        print("Epoch: {}, | Training loss: {:.5f}  | Validation Loss: {:.5f} |  Test Loss: {:.5f} | "
              "Training Accuracy: {:.5f}  | Validation Accuracy: {:.5f} |  Test Accuracy: {:.5f}"
              .format(epoch + 1, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc))

        if np.linalg.norm(grad_weights) <= EPS or np.linalg.norm(grad_biases) <= EPS:
           break

    elapsed_time = int(time.time() - start_time)

    print("Training Complete, Time taken is: {:02d}:{:02d}:{:02d}".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60))

    return W, b





def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=0.001):

    # Initialize weight and bias tensors
    weights = tf.truncated_normal(W.shape, stddev=0.5)
    weights = np.reshape(weights, (weights.shape[0] * weights.shape[1], -1))
    bias = 0
    data = tf.placeholder(tf.float64, trainData.shape)
    labels = tf.placeholder(tf.int8, trainTarget.shape)
    regularization = tf.placeholder(tf.float64, (1,1))


    tf.set_random_seed(421)
    if loss == "MSE":
        loss_tensor = MSE(W=weights, b=bias, x=data, y=labels, reg=regularization)
        predicted_labels = tf.matmul(trainingData, weights) + b
    elif loss == "CE":
        loss_tensor = crossEntropyLoss(W=weights, b=bias, x=data, y=labels, reg=regularization)
        predicted_labels = tf.sigmoid(tf.matmul(trainingData, weights))

    opt = GradientDescentOptimizer(learning_rate=learning_rate)
    opt_op = opt.minimize(loss_tensor, var_list=[W, b])

    return weights, bias, predicted_labels, labels, loss_tensor, opt_op, regularization

test_normal = False
test_GD = True

if test_GD:
    W, b = grad_descent(W, b, trainData, trainTarget, lrs[0], epochs, reg[0], error_tolerance, 'CE')
    plot(epochs, trainloss_list, valloss_list, testloss_list, train_acc_list, val_acc_list, test_acc_list, True)

if test_normal:
    w_least_squares = WLS(trainData, trainTarget, reg[0])
    analytical_loss = MSE(w_least_squares, 0, trainData.reshape((trainData.shape[0], -1)), trainTarget, reg[0])
    print("Final training MSE for normal equation: {:02f} ".format(analytical_loss))

