'''
ECE421 A3 part 2

Code written by Andy Zhou, Ryan Do

Mar 19, 2019
'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import os
import math
import random
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
total_data = data
is_valid = False
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

    na = tf.reduce_sum(tf.square(X), 1)
    nb = tf.reduce_sum(tf.square(MU), 1)

    # na is N x K, nb is N x K
    # broadcast na along columns, broadcast nb along rows
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = na - 2 * tf.matmul(X, MU, False, True) + nb

    return D



def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1
    # D: N X K

    # Outputs:
    # log Gaussian PDF N X K

    # 2.1.1 Log PDF for cluster k

    # compute pairwise squared distance as in part 1.1

    na = tf.reduce_sum(tf.square(X), 1)
    nb = tf.reduce_sum(tf.square(mu), 1)

    # na is N x K, nb is N x K
    # broadcast na along columns, broadcast nb along rows
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = na - 2 * tf.matmul(X, mu, False, True) + nb     # N x K

    # square to get variances, then reshape into row tensor
    sigma2_mat = tf.ones([tf.shape(X)[0], 1], dtype=tf.float64) * tf.reshape(sigma, [1, -1])

    coeff_1 = -(tf.shape(X)[1]/2) * tf.log(2*tf.cast(np.pi, dtype=tf.float64))       # 1 X 1

    coeff_2 = -(1/2)* tf.log(tf.pow(sigma, tf.cast(tf.fill([tf.shape(sigma)[0], 1], tf.shape(X)[1]), dtype=tf.float64)))  # K x 1

    coeff = coeff_1 + coeff_2

    mat = -(1/2) * tf.multiply(tf.math.reciprocal(sigma2_mat), D)

    return tf.squeeze(coeff) + mat




def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # part 2.2
    # log P(Z = k| x) = log P(x|mu, sigma) + log_pi - logsumexp(z)


    term1 = tf.squeeze(log_pi) + log_PDF    # N x K

    log_post = term1 - tf.expand_dims(hlp.reduce_logsumexp(term1), 1)

    return log_post


def NLL_loss(log_gauss_PDF, log_pi):


    inner = tf.squeeze(log_pi) + log_gauss_PDF

    summed_over_clusters = hlp.reduce_logsumexp(inner)

    loss = -tf.reduce_sum(summed_over_clusters)


    return loss





num_updates = 800
lr = 0.01
K = 3




x = tf.placeholder(name='x', dtype=tf.float64, shape=(None, data.shape[1]))
mu = tf.get_variable(name='mean_vector', dtype=tf.float64, shape=(K, data.shape[1]), initializer=tf.initializers.random_normal(seed=0))
phi = tf.get_variable(name='stdev_vector', dtype=tf.float64, shape=(K, 1), initializer=tf.initializers.random_normal(seed=0))
psi = tf.get_variable(name='pi_vector', dtype=tf.float64, shape=(K, 1), initializer=tf.initializers.random_normal(seed=0))

sigma = tf.exp(phi)

logGaussPDF = log_GaussPDF(x, mu, sigma)
logPi = hlp.logsoftmax(psi)

log_post = log_posterior(logGaussPDF, logPi)

NLLloss = NLL_loss(logGaussPDF, logPi)

optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(NLLloss)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train_loss_list = []

    for i in range(num_updates):
        sess.run(optimizer, feed_dict={x: data})

        train_loss = sess.run(NLLloss, feed_dict={x: data})

        if is_valid:
            val_loss = sess.run(NLLloss, feed_dict={x: val_data})
            print('The training loss is: {} | The validation loss is: {}'.format(train_loss, round(val_loss, 2)))
        else:
            print('The training loss is: {} '.format(round(train_loss, 2)))

        train_loss_list.append(train_loss)



    final_mu = sess.run(mu)
    final_sigma = sess.run(sigma)
    final_posterior = sess.run(log_post, feed_dict={x: total_data})
    final_pi = np.exp(sess.run(logPi, feed_dict={x: total_data}))

    #print('the final mu is: ', np.around(final_mu, decimals=3))
    print('the final sigma is: ', np.around(final_sigma, decimals=3))
    print('the final pi is: ', np.around(final_pi, decimals=3))


    loss_curve = plt.plot(train_loss_list)
    loss_curve = plt.xlabel('Number of updates')
    loss_curve = plt.ylabel('Loss')
    loss_curve = plt.title("K={}, Loss VS Number of updates (MoG)".format(K))

    plt.show(loss_curve)


    
    data_cluster_mat = np.column_stack((total_data, np.ones((total_data.shape[0], 1))))

    for i, point in enumerate(data_cluster_mat):
        probabilities = final_posterior[i]
        point[-1] = np.argmax(probabilities) + 1

    unique, counts = np.unique(data_cluster_mat[:, -1], return_counts=True)
    dict_counts = dict(zip(unique, counts))
    #print(dict_counts)


    
    for cluster in range(1, K + 1):

        try:
            percentage = dict_counts[cluster] * 100 / total_data.shape[0]
            print('The percentage of points belonging to cluster {} is: {}% '.format(cluster, percentage))

        except KeyError:
            print('Cluster {} has no points belonging to it'.format(cluster))


    try:
        x_mu, y_mu = final_mu.T
        x, y, cluster_label = data_cluster_mat.T


        for g in np.unique(cluster_label):
            i = np.where(cluster_label == g)
            plt.scatter(x[i], y[i], label='Cluster ' + str(int(g)), s=8)

        #plt.scatter(x, y, c=cluster_label, label='data')
        plt.scatter(x_mu, y_mu, cmap='r', marker='X', label='centroids', c='k')

        n = [str(i) for i in range(1, K+1)]

        for i, txt in enumerate(n):
            plt.annotate(txt, (x_mu[i], y_mu[i]))

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Result of running Gaussian Mixture Algorithm with K = {}'.format(K))
        plt.legend()
        plt.show()

    except:
        print("Currently testing 100 Dimensional DataSet")





