import numpy as np
import pickle


def compute_sousep(x, y, m, num_sources, fname=None, use_sampled_cov=False):
    """compute source separation."""

    if use_sampled_cov:
        # matrix_path = '/import/c4dm-04/alvarado/results/sampling_covariance/'
        matrix_path = '/home/pa/Desktop/sampling_covariance/'
        pitches = ["60", "64", "67"]
        cov = []
        var1 = m.kern.prod_1.matern12.variance.value[0]
        var2 = m.kern.prod_2.matern12.variance.value[0]
        var3 = m.kern.prod_3.matern12.variance.value[0]
        var = [var1, var2, var3]

        for i in range(num_sources):
            path = matrix_path + fname + '_M' + pitches[i] + "_cov_matrix.p"

            cov.append(var[i] * pickle.load(open(path, "rb")))
        K = sum(cov)
        Ks = list(cov)

    else:
        K = m.kern.compute_K_symm(x)  # inverse of full covariance
        Ks1 = m.kern.prod_1.compute_K_symm(x)  # covariance for each source
        Ks2 = m.kern.prod_2.compute_K_symm(x)
        Ks3 = m.kern.prod_3.compute_K_symm(x)
        Ks = [Ks1, Ks2, Ks3]

    noise_var = m.likelihood.variance.value[0]
    L = np.linalg.cholesky(K + noise_var * np.eye(x.size))
    Linv = np.linalg.inv(L)
    Kinv = np.matmul(Linv.T, Linv)

    yhat = np.matmul(Kinv, y)  # product needed for each source
    esources = []  # list estimated sources

    for i in range(num_sources):
        esources.append(np.matmul(Ks[i], yhat))

    return esources