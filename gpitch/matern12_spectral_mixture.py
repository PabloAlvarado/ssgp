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


class Matern12sm(gpflow.kernels.Kern):
    """
    Matern spectral mixture kernel with single lengthscale.
    """
    def __init__(self, input_dim,  variance=1., lengthscales=None, energy=None, frequencies=None,
                 len_fixed=True):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        energy_l = []
        freq_l = []
        self.ARD = False
        self.num_partials = len(energy)

        for i in range(self.num_partials):
            energy_l.append(Param(energy[i], transforms.positive))
            freq_l.append(Param(frequencies[i], transforms.positive))

        self.energy = ParamList(energy_l)
        self.frequency = ParamList(freq_l)
        self.variance = Param(variance, transforms.positive)
        self.lengthscales = Param(lengthscales, transforms.positive)

        self.vars_n_freqs_fixed(fix_energy=True, fix_freq=True)
        if len_fixed:
            self.lengthscales.fixed = True

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.sqrt(tf.square(f - f2 +  1e-12))

        r1 = tf.reduce_sum(r / self.lengthscales, 2)
        r2 = tf.reduce_sum(2.*np.pi * self.frequency[0] * r, 2)
        k = self.energy[0] * tf.cos(r2)

        for i in range(1, self.num_partials):
            r2 = tf.reduce_sum(2.*np.pi*self.frequency[i]*r, 2)
            k += self.energy[i] * tf.cos(r2)
        return self.variance * tf.exp(-r1) * k

    def Kdiag(self, X):
        var = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.energy[0]))
        for i in range(1, self.num_partials):
            var += tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze( self.energy[i]))
        return self.variance * var

    def vars_n_freqs_fixed(self, fix_energy=True, fix_freq=True):
        for i in range(self.num_partials):
            self.energy[i].fixed = fix_energy
            self.frequency[i].fixed = fix_freq
