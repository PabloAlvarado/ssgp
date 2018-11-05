import numpy as np
import scipy as sp


def sample_cov(x, niter, msize):
    """Infer covariance matrix by sampling segments from a large vector"""
    cov = np.zeros((msize, msize))
    samples = niter*[None]
    for i in range(niter):
        idx = np.random.randint(0, x.size - msize)  # sample start index to get audio segment of sise msize
        samples[i] = x[idx: idx + msize].copy()
        # if np.random.randint(0, 2):
        #     samples[i] = np.flipud(samples[i].copy())
        cov += np.outer(samples[i], samples[i])
    cov/(1.*niter)  # get mean matrix
    cov /= np.max(cov)  # scaled between (-1, 1)
    lower = np.linalg.cholesky(cov + 0.000001*np.eye(msize))  # be sure it is positive semi-definite
    return np.matmul(lower, lower.T), samples


def loss_func(p, x, y):
    '''
    Loss function to fit function to kernel observations
    '''
    f =  np.sqrt(np.square(approximate_kernel(p, x) - y).mean())
    return f


def approximate_kernel(p, x):
    '''
    approximate kernel
    '''
    nparams = p.size
    npartials = (nparams - 2) / 2
    bias = np.sqrt(p[0] * p[0])
    # k_e = (1. + np.sqrt(3.) * np.abs(x) / np.sqrt(p[1] * p[1])) * np.exp(- np.sqrt(3.) * np.abs(x) / np.sqrt(p[1] * p[1]))
    k_e = np.exp(- np.abs(x) / np.sqrt(p[1] * p[1]))

    k_partials = [np.sqrt(p[i] * p[i]) * np.cos(2 * np.pi * np.sqrt(p[i + npartials] * p[i + npartials]) * np.abs(x))
                  for i in range(2, 2 + npartials)]
    k_fun = 0.*bias + k_e * sum(k_partials)
    #     nparams = p.size
    #     npartials = (nparams - 1)/3
    #     bias = np.sqrt(p[0]*p[0])
    #     k_all = [( 1. + np.sqrt(3.) * x/np.sqrt(p[i]*p[i]) ) * np.exp( - np.sqrt(3.) * x / np.sqrt(p[i]*p[i]) ) *
    #              np.sqrt(p[i + npartials]*p[i + npartials]) *
    #              np.cos(2*np.pi * np.sqrt(p[i + 2*npartials]*p[i + 2*npartials]) * x)
    #              for i in range(1, npartials + 1)]
    #     k_fun = bias + sum(k_all)
    return k_fun


def optimize_kern(x, y, p0):
    """Optimization of kernel"""
    phat = sp.optimize.minimize(loss_func, p0, method='L-BFGS-B', args=(x, y), tol=1e-12, options={'disp': True})
    pstar = np.sqrt(phat.x ** 2).copy()
    return pstar





