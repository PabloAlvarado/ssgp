import numpy as np
import tensorflow as tf


def get_samples(x, num_sam, size):
    """
    Infer covariance matrix by sampling segments from a large vector
    """
    samples = num_sam * [None]
    for i in range(num_sam):
        idx = np.random.randint(0, x.size - size)  # sample start index to get audio segment of sise msize
        samples[i] = x[idx: idx + size].copy()
    return samples


def comatrix(X):
    """
    Compute the approximate covariance matrix using tensorflow.
    """
    #  get number of samples and vector size
    num_sam = len(X)
    vec_size = X[0].size

    #  init cov matrix
    cov = np.zeros((vec_size, vec_size))

    # define graph
    vec = tf.placeholder(dtype=tf.float64)  # vector to compute outer product
    outer = tf.matmul(vec, vec, transpose_b=True)

    with tf.Session() as sess:
        for i in range(num_sam):
            cov += sess.run(outer, feed_dict={vec: X[i]})
    return (1./num_sam)*cov


def get_cov(x, num_sam, size):
    """
    :param x:
    :param num_sam:
    :param size:
    :return:
    covariance matrix, samples used, kernel
    """
    # samples = lenglist*[None]
    # cov = lenglist*[None]
    # kern = lenglist*[None]

    samples = get_samples(x, num_sam, size)
    cov = comatrix(samples)
    kern = cov[0, :].copy().reshape(-1, 1)
    kern /= np.max(np.abs(kern))
    return cov, kern, samples


def autocorr(x, size):
    """
    Infer covariance matrix by sampling segments from a large vector
    """
    num_sam = x.size - size
    samples = num_sam * [None]
    for i in range(num_sam):
        idx = i
        samples[i] = x[idx: idx + size].copy()
        
    samples = np.asarray(samples)
    samples =  np.squeeze(samples, 2).T
    
    r = np.zeros((samples.shape[0], ))
    for i in range(samples.shape[1]):
        r += samples[0, i].copy() * samples[:,i].copy()
    r /= np.max(np.abs(r))
    
    return r.reshape(-1, 1), samples
