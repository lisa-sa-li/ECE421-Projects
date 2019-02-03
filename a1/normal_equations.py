import time
import numpy as np




def WLS(trainData, trainTarget, reg):


    x = trainData
    x = np.reshape(x, (x.shape[0], -1))
    y = trainTarget

    start_time = time.time()

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x) + np.identity(x.shape[1])*reg), x.T), trainTarget)

    elapsed_time = int(start_time - time.time())

    predicted_train = np.matmul(x, w)

    predicted_train[predicted_train > 0] = 1
    predicted_train[predicted_train < 0] = 0

    train_acc = np.sum(predicted_train == trainTarget) / x.shape[0]


    print("The accuracy of the analytical Least Squares solution is: {:02f}".format(train_acc))

    print("Training Complete, Time taken is: {:02d}:{:02d}:{:02d}".format(elapsed_time // 3600,
                                                                          (elapsed_time % 3600 // 60),
                                                                          elapsed_time % 60))

    return w