import numpy as np
import tensorflow as tf
import gpflow
from gpflow.param import ParamList, Param, transforms
from gpflow import settings
from scipy.signal import hann

float_type = settings.dtypes.float_type
jitter = settings.numerics.jitter_level

int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Env(gpflow.kernels.Kern):
    """Envelope kernel"""

    def __init__(self, input_dim, z):
        gpflow.kernels.Kern.__init__(self, input_dim=input_dim, active_dims=None)

        self.kernel = gpflow.kernels.RBF(input_dim=input_dim, lengthscales=0.1, variance=0.25)
        #self.kernel.variance.fixed = True
        self.z = gpflow.param.Param(z)
        self.u = gpflow.param.Param(0.*np.sqrt(0.001)*np.random.randn(z.size, 1))

    def build_function(self, X):
        
        K = self.kernel.K(self.z) + 0.001*tf.eye(tf.shape(self.z)[0], dtype=float_type)
        Kx = self.kernel.K(self.z, X)
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.u)
        g = tf.matmul(A, V, transpose_a=True)
        
        return tf.log( 1. + tf.exp(g))

    def K(self, X, X2=None, presliced=False):

        if not presliced:
            X, X2 = self._slice(X, X2)

        if X2 is None:
            Xhat = self.build_function(X)
            return tf.matmul(Xhat, Xhat, transpose_b=True)

        else:
            Xhat = self.build_function(X)
            Xhat2 = self.build_function(X2)
            return tf.matmul(Xhat, Xhat2, transpose_b=True)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        Xhat = self.build_function(X)
        return tf.reduce_sum(tf.square(Xhat), 1)


class Sig(gpflow.kernels.Kern):
    """
    The sigmoidal kernel with unitary variance.
    """

    def __init__(self, input_dim, a=1.0, b=1.0, active_dims=None):
        """
        """
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims)

        self.a = Param(a)
        self.b = Param(b)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            Xhat = 1. / (1. + tf.exp(-(X * self.a + self.b)))
            return tf.matmul(Xhat, Xhat, transpose_b=True)
        else:
            Xhat = 1. / (1. + tf.exp(-(X * self.a + self.b)))
            Xhat2 = 1. / (1. + tf.exp(-(X2 * self.a + self.b)))
            return tf.matmul(Xhat, Xhat2, transpose_b=True)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        Xhat = 1. / (1. + tf.exp(-(X * self.a + self.b)))
        return tf.reduce_sum(tf.square(Xhat), 1)


class Hann(gpflow.kernels.Kern):
    """
    The Hanning kernel with unitary variance.
    """

    def __init__(self, input_dim, N=1025, active_dims=None):
        """
        """
        self.N = N
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            Xhat = 0.5 * (1. - tf.cos(2.*np.pi*X*16000/(self.N - 1.)) )
            return tf.matmul(Xhat, Xhat, transpose_b=True)
        else:
            Xhat = 0.5 * (1. - tf.cos(2. * np.pi * X * 16000 / (self.N - 1.)))
            Xhat2 = 0.5 * (1. - tf.cos(2. * np.pi * X2 * 16000 / (self.N - 1.)))
            return tf.matmul(Xhat, Xhat2, transpose_b=True)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        Xhat = 0.5 * (1. - tf.cos(2.*np.pi*X*16000/(self.N - 1.)) )
        return tf.reduce_sum(tf.square(Xhat), 1)


class Cosine(gpflow.kernels.Kern):
    """
    The Cosine kernel with frequency hyperparameter, instead of lengthscale
    """

    def __init__(self, input_dim, variance=1., frequency=1.):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.variance = Param(variance, transforms.positive)
        self.frequency = Param(frequency, transforms.positive)

    def square_dist(self, X, X2):
        X = 2. * np.pi * self.frequency * X
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = 2. * np.pi * self.frequency * X2
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.cos(r)

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class Matern32sm_old(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim, numc, lengthscales=None, variances=None, frequencies=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.ARD = False
        self.numc = numc

        if lengthscales == None:
            lengthscales = 1.
            variances = 0.125*np.ones((numc, 1))
            frequencies = 1.*np.arange(1, numc+1)

        self.lengthscales = Param(lengthscales, transforms.Logistic(0., 10.) )
        for i in range(self.numc): # generate a param object for each  var, and freq, they must be (numc,) arrays.
            setattr(self, 'variance_' + str(i+1), Param(variances[i], transforms.Logistic(0., 0.25) ) )
            setattr(self, 'frequency_' + str(i+1), Param(frequencies[i], transforms.positive ) )

        for i in range(self.numc):
            exec('self.variance_' + str(i + 1) + '.fixed = ' + str(True))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = np.sqrt(3.)*tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency_1 * r , 2)
        k = self.variance_1 * (1. + r1) * tf.exp(-r1) * tf.cos(r2)

        for i in range(2, self.numc + 1):
            r2 = tf.reduce_sum(2.*np.pi * getattr(self, 'frequency_' + str(i)) * r , 2)
            k += getattr(self, 'variance_' + str(i)) * (1. + r1) * tf.exp(-r1) * tf.cos(r2)
        return k


    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance_1))
        for i in range(2, self.numc + 1):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(getattr(self, 'variance_' + str(i))))
        return var


class Matern32sm(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim, num_partials, lengthscales=None, variances=None, frequencies=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        var_l = []
        freq_l = []
        self.ARD = False
        self.num_partials = num_partials

        if lengthscales == None:
            lengthscales = 1.
            variances = 0.125*np.ones((num_partials, 1))
            frequencies = 1.*(1. + np.arange(num_partials))

        self.lengthscales = Param(lengthscales, transforms.Logistic(0., 2.))

        for i in range(self.num_partials):
            var_l.append(Param(variances[i], transforms.Logistic(0., 0.25)))
            freq_l.append(Param(frequencies[i], transforms.positive))

        self.variance = ParamList(var_l)
        self.frequency = ParamList(freq_l)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = np.sqrt(3.)*tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency[0] * r , 2)
        k = self.variance[0] * (1. + r1) * tf.exp(-r1) * tf.cos(r2)

        for i in range(1, self.num_partials):
            r2 = tf.reduce_sum(2.*np.pi*self.frequency[i]*r , 2)
            k += self.variance[i] * (1. + r1) * tf.exp(-r1) * tf.cos(r2)
        return k

    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance[0]))
        for i in range(1, self.num_partials):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze( self.variance[i] ) )
        return var

    def vars_n_freqs_fixed(self, fix_var=True, fix_freq=False):
        for i in range(self.num_partials):
            self.variance[i].fixed = fix_var
            self.frequency[i].fixed = fix_freq


class Matern32sml(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim, num_partials, lengthscales=None, variances=None, frequencies=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        len_l = []
        var_l = []
        freq_l = []
        self.ARD = False
        self.num_partials = num_partials

        if lengthscales.all() == None:
            lengthscales = 1.*np.ones((num_partials, 1))
            variances = 0.125*np.ones((num_partials, 1))
            frequencies = 1.*(1. + np.arange(num_partials))

        for i in range(self.num_partials):
            len_l.append(Param(lengthscales[i], transforms.Logistic(0., 2.)))
            var_l.append(Param(variances[i], transforms.Logistic(0., 1.)))
            freq_l.append(Param(frequencies[i], transforms.positive))

        self.lengthscales = ParamList(len_l)
        self.variance = ParamList(var_l)
        self.frequency = ParamList(freq_l)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = np.sqrt(3.)*tf.reduce_sum(r / self.lengthscales[0], 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency[0] * r , 2)
        k = self.variance[0] * (1. + r1) * tf.exp(-r1) * tf.cos(r2)

        for i in range(1, self.num_partials):
            r1 = np.sqrt(3.)*tf.reduce_sum(r / self.lengthscales[i], 2)
            r2 = tf.reduce_sum(2.*np.pi*self.frequency[i]*r , 2)
            k += self.variance[i] * (1. + r1) * tf.exp(-r1) * tf.cos(r2)
        return k

    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance[0]))
        for i in range(1, self.num_partials):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze( self.variance[i] ) )
        return var

    def vars_n_freqs_fixed(self, fix_len = False, fix_var=False, fix_freq=False):
        for i in range(self.num_partials):
            self.variance[i].fixed = fix_var
            self.frequency[i].fixed = fix_freq
            self.lengthscales[i].fixed = fix_len


class MercerCosMix(gpflow.kernels.Kern):
    """
    The Mercer Cosine Mixture kernel for audio.
    """

    def __init__(self, input_dim, energy=np.asarray([1.]), frequency=np.asarray([2*np.pi]),
                 variance=1.0, features_as_params=False):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.num_features = len(frequency)
        self.variance = Param(variance, transforms.Logistic(0., 0.25))

        if features_as_params:
            energy_list = []
            frequency_list = []
            for i in range(energy.size):
                energy_list.append( Param(energy[i], transforms.positive) )
                frequency_list.append( Param(frequency[i], transforms.positive) )

            self.energy = ParamList(energy_list)
            self.frequency = ParamList(frequency_list)
        else:
            self.energy = energy
            self.frequency = frequency

    def phi_features(self, X):
        n = tf.shape(X)[0]
        m = self.num_features
        phi_list = 2*m*[None]
        for i in range(m):
            phi_list[i] = tf.sqrt(self.energy[i])*tf.cos(2*np.pi*self.frequency[i]*X)
            phi_list[i + m] = tf.sqrt(self.energy[i])*tf.sin(2*np.pi*self.frequency[i]*X)
        phi = tf.stack(phi_list)
        return tf.reshape(phi, (2*m, n))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            phi = self.phi_features(X)
            k = tf.matmul(phi * self.variance, phi, transpose_a=True)
            return k
        else:
            phi = self.phi_features(X)
            phi2 = self.phi_features(X2)
            k = tf.matmul(phi * self.variance, phi2, transpose_a=True)
            return k

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class Logistic_hat(gpflow.kernels.Stationary):
    """
    The Logistic hat kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        f1 = (1./ (1. + tf.exp( 100*(-1.-r) )) )
        f2 = (1./ (1. + tf.exp( 100*( 1.-r) )) )
        return self.variance * (f1 - f2)


class Spectrum(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim, frequency=None, energy=None, variance=1.0):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)

        self.ARD = False
        self.num_partials = len(frequency)

        self.energy = energy
        self.variance = Param(variance, transforms.positive)
        self.frequency = frequency

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        k_list = self.num_partials*[None]
        for i in range(self.num_partials):
            r2 = tf.reduce_sum(2.*np.pi*self.frequency[i]*r , 2)
            k_list[i] = self.energy[i] * tf.cos(r2)
        k = tf.reduce_sum(k_list, 0)

        return self.variance*k

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class Spectrum2(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim, frequency=None, energy=None, variance=1.0):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)

        self.ARD = False
        self.num_partials = len(frequency)

        self.energy = energy
        self.variance = Param(variance, transforms.positive)
        self.frequency = frequency

    def square_dist_2(self, X, X2):
        X = X
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist_2(self, X, X2, freq):
        r2 = self.square_dist_2(X, X2)
        return 2.*np.pi*freq*tf.sqrt(r2 + 1e-12)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        k_list = self.num_partials*[None]
        for i in range(self.num_partials):
            r = self.euclid_dist_2(X, X2, self.frequency[i])
            k_list[i] = self.energy[i] * tf.cos(r)
        k = tf.reduce_sum(k_list, 0)

        return self.variance*k

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class NonParam(gpflow.kernels.Kern):
    """Non-parametric kernel"""

    def __init__(self, input_dim, numsamples, variance=1.0):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.ARD = False
        self.numsamples = numsamples

        self.variance = gpflow.param.Param(variance, transform=gpflow.transforms.positive)

        self.L = gpflow.param.Param(np.eye(self.numsamples),
                                    transform=gpflow.transforms.LowerTriangular(N=self.numsamples))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return tf.squeeze(tf.matmul(self.L * self.variance, self.L, transpose_b=True))

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class MeanGP(gpflow.kernels.Stationary):
    def __init__(self, input_dim, xkern, fkern, variance=1.0, plen=1.0, pvar=1.0):

        gpflow.kernels.Stationary.__init__(self, input_dim=input_dim, active_dims=None, ARD=False)
        eyem = tf.eye(xkern.size, dtype=float_type)
        self.variance = Param(variance, transforms.positive)
        self.plen = plen
        self.pvar = pvar
        self.fkern = fkern
        self.xkern = xkern
        self.kern = gpflow.kernels.RBF(input_dim=input_dim, variance=self.pvar, lengthscales=self.plen)
        self.cov = tf.matmul(eyem, self.kern.compute_K_symm(xkern))
        self.icov = tf.matrix_inverse(self.cov + jitter*eyem)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        K_fu = self.kern.compute_K(tf.reshape(r, (-1, 1)), self.xkern)
        return K_fu


class KernelGPR(gpflow.kernels.Kern):
    """
    The GP kernel
    """
    def __init__(self, input_dim, gpm, variance=1.0):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.variance = Param(variance, transforms.positive)
        self.m = gpm
        self.m.fixed = True

    def square_dist(self, X, X2):
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def K(self, X, X2=None, presliced=False):

        if not presliced:
            X, X2 = self._slice(X, X2)

        r = self.euclid_dist(X, X2)
        cov = self.m.build_predict(tf.reshape(r, (-1, 1)))[0]
        return self.variance * tf.reshape(cov, (tf.shape(r)[0], tf.shape(r)[1]))

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.m.kern.variance * self.variance))


class Gammaexponential(gpflow.kernels.Stationary):
    """
    The Exponential kernel
    """
    def __init__(self, input_dim, variance=1., lengthscales=1., gamma=1.):
        gpflow.kernels.Stationary.__init__(self, input_dim=input_dim, variance=variance, lengthscales=lengthscales)
        self.gamma = Param(gamma, transforms.Logistic(0.00001, 2.))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-r**self.gamma)























"""end"""
