'''
ECE421 A3 part 1

Code written by Andy Zhou, Ryan Do

Mar 19, 2019

'''




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
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


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

    na = tf.reduce_sum(tf.square(X), 1)
    nb = tf.reduce_sum(tf.square(MU), 1)

    # na = tf.tile(tf.expand_dims(na, -1), [1, tf.shape(mu)[0]])

    # na is N x K, nb is N x K
    # broadcast na along columns, broadcast nb along rows
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = na - 2 * tf.matmul(X, MU, False, True) + nb


    return D


num_updates = 600
lr = 0.01
K = 3


def k_means(num_updates, lr, K, data):


    x = tf.placeholder(name='x', dtype=tf.float64, shape=(None, data.shape[1]))
    mu = tf.get_variable(name='mean_vector', dtype=tf.float64, shape=(K, data.shape[1]), initializer=tf.initializers.random_normal(seed=0))

    #mu = tf.random_normal(shape=(K, data.shape[1]))

    loss = tf.reduce_sum(tf.reduce_min(distanceFunc(x, mu), axis=1))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_loss_list = []

        for i in range(num_updates):

            sess.run(optimizer, feed_dict={x:data})

            train_loss = sess.run(loss, feed_dict={x:data})

            if is_valid:
                val_loss = sess.run(loss, feed_dict={x:val_data})

                print('The training loss is: {} | The validation loss is: {} '.format(train_loss, round(val_loss, 2)))

            else:
                print('The training loss is: {}'.format(round(train_loss, 2)))


            train_loss_list.append(train_loss)


        final_mu = sess.run(mu)
        loss_curve = plt.plot(train_loss_list)
        loss_curve = plt.xlabel('Number of updates')
        loss_curve = plt.ylabel('Loss')
        loss_curve = plt.title("K={}, Loss VS Number of updates (K-Means)".format(K))


        plt.show(loss_curve)

        #print(final_mu)

        data_cluster_mat = np.column_stack((total_data, np.ones((total_data.shape[0], 1))))

        for point in data_cluster_mat:


            distances = np.array([np.linalg.norm(point[:-1]-center) for center in final_mu])
            point[-1] = np.argmin(distances)+1


        unique, counts = np.unique(data_cluster_mat[:, -1], return_counts=True)
        dict_counts = dict(zip(unique, counts))

        for cluster in range(1,K+1):

            try:
                percentage = dict_counts[cluster] * 100/total_data.shape[0]
                print('The percentage of points belonging to cluster {} is: {}% '.format(cluster, round(percentage, 2)))

            except KeyError:
                print('Cluster {} has no points belonging to it'.format(cluster))

        try:
            x_mu, y_mu = final_mu.T
            x, y, cluster_label = data_cluster_mat.T

            for g in np.unique(cluster_label):
                i = np.where(cluster_label == g)
                plt.scatter(x[i], y[i], label='Cluster ' + str(int(g)), s=8)

            #plt.scatter(x, y, c=cluster_label, label='data')
            plt.scatter(x_mu, y_mu, cmap='r', marker='X', label='centroids', c='navy')

            n = [str(i) for i in range(1, K + 1)]

            for i, txt in enumerate(n):
                plt.annotate(txt, (x_mu[i], y_mu[i]))


            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Result of running K-means algorithm with K = {}'.format(K))
            plt.legend()
            plt.show()

        except:
            print("Currently testing 100 Dimensional DataSet")

        return final_mu




mu = k_means(num_updates, lr, K, data)
#np.save('mu_3', mu_3)