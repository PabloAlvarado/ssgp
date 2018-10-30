import gpflow
import tensorflow as tf
from gpflow.param import AutoFlow, DataHolder
from gpflow import settings


float_type = settings.dtypes.float_type


class SGPRSS(gpflow.sgpr.SGPR):
    """
    Sparse Gaussian process regression for source separation
    """
    def __init__(self, X, Y, kern, Z, mean_function=None):
        gpflow.sgpr.SGPR.__init__(self, X=X, Y=Y, kern=kern, Z=Z, mean_function=mean_function)
        self.Z = DataHolder(Z, on_shape_change='pass')

    def build_predict_source(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(source* | Y )

        where source* are points on the source at Xnew, Y are noisy observations of the mixture at X.

        """
        mean = []
        var = []
        nsources = len(self.kern.kern_list)

        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))

        for i in range(nsources):
            Kx = self.kern.kern_list[i].K(self.X, Xnew)
            A = tf.matrix_triangular_solve(L, Kx, lower=True)
            smean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
            if full_cov:
                svar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
                shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
                svar = tf.tile(tf.expand_dims(svar, 2), shape)
            else:
                svar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
                svar = tf.tile(tf.reshape(svar, (-1, 1)), [1, tf.shape(self.Y)[1]])

            mean.append(smean)
            var.append(svar)
        return mean, var

    @AutoFlow((float_type, [None, None]))
    def predict_s(self, Xnew):
        """
        Compute the mean and variance of the sources
        at the points `Xnew`.
        """
        return self.build_predict_source(Xnew)