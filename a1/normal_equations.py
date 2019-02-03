from starter import loadData
import numpy as np
import os

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

reg = 0
x = trainData
x = np.reshape(x, (x.shape[0], -1))
y = trainTarget

w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x) + np.identity(x.shape[1])*reg), x.T), trainTarget)

predicted_train = np.matmul(x, w)

predicted_train[predicted_train > 0] = 1
predicted_train[predicted_train < 0] = 0

train_acc = np.sum(predicted_train == trainTarget) / x.shape[0]


print(train_acc)