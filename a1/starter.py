import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

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
    bceloss = -y * np.log(logistic_y_hat(W, x, b)) - (1 - y) * np.log(1 - logistic_y_hat(W, x, b))
    weightdecay = reg / 2 * np.linalg.norm(W) ^ 2
    return (bceloss / x.shape[0]).cumsum(axis=0) + weightdecay

def gradCE(W, b, x, y, reg):
    grad_w = ((-y*x + x*logistic_y_hat(W, x, b))/x.shape[0]).cumsum(axis=0) + reg*W
    # see derivation: https://github.com/Exquisition/ECE421-Projects/blob/master/a1/bceloss_gradient_derivation.jpg
    # Can also verify using tf.gradients on w
    grad_b = tf.gradients(crossEntropyLoss(W, b, x, y, reg), b)
    return [grad_w, grad_b]

def logistic_y_hat (W, x, b):
    return tf.math.sigmoid(np.transpose(W) * x + b)

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
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
        gradients = gradMSE(W, b, trainingData, trainingLabels, reg)

        grad_weights = gradients[0]
        grad_biases = gradients[1]

        W = W - alpha*grad_weights
        b = b - alpha*grad_biases

        train_loss = MSE(W, b, trainingData, trainingLabels, reg)
        val_loss = MSE(W, b, validData, validTarget, reg)

        predicted_train = np.sign(np.matmul(trainingData, W) + b)
        predicted_val = np.sign(np.matmul(validData, W) + b)

        train_acc = np.sum(predicted_train == trainingLabels) / N
        val_acc = np.sum(predicted_val == validTarget) / N

        print("Epoch: {}, | Training loss: {}  | Validation Loss: {} | Training Accuracy: {}  | Validation Accuracy: {}"
              .format(epoch, train_loss, val_loss, train_acc, val_acc))

        if np.linalg.norm(grad_weights) <= EPS or np.linalg.norm(grad_biases) <= EPS:
            break

    elapsed_time = int(time.time() - start_time)

    print("Training Complete, Time taken is: {:02d}:{:02d}:{:02d}".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60))

    return W, b





def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass





W, b = grad_descent(W, b, trainData, trainTarget, lrs[0], epochs, reg[1], error_tolerance)
print(b)