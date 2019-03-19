import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from starter_kmeans import *

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

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


    sigma2_mat = tf.linalg.diag(tf.square(tf.transpose(sigma)))      # this is a K x K covariance matrix
    sigma2_mat = tf.squeeze(sigma2_mat)


    coeff = -(X.shape[1]/2)*tf.cast(tf.log(2*math.pi), tf.float64) - (1/2)*tf.log(tf.linalg.det(sigma2_mat))


    mat = -(1/2) * tf.matmul(D, tf.linalg.inv(sigma2_mat))

    return coeff + mat




def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # part 2.2
    # log P(Z = k| x) = log P(x|mu, sigma) + log_pi - logsumexp(z)



    log_post = log_PDF + tf.squeeze(log_pi) - tf.expand_dims(hlp.reduce_logsumexp(log_PDF), 1)

    return log_post


def NLL_loss(log_post, log_gauss_PDF):

    pass





num_updates = 600
lr = 0.01
K = 3
mu_3 = np.load('mu_3.npy')
print(mu_3.shape)

#mean = k_means(num_updates, lr, K, data)
st_devs = tf.random_normal([K, 1], dtype=tf.float64)

log_norm_PDF = log_GaussPDF(data, mu_3, st_devs)
log_pos = log_posterior(log_norm_PDF, st_devs)

