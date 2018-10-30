import numpy as np
import tensorflow as tf
import gpitch
import gpflow
from gpflow import settings
from gpflow.minibatch import MinibatchData
from likelihoods import MpdLik
from gpflow.param import Param, ParamList
from gpflow.kullback_leiblers import gauss_kl
from gpitch.methods import logistic_tf, gaussfun_tf


float_type = settings.dtypes.float_type
jitter = settings.numerics.jitter_level


def predict_windowed(model, xnew, ws=1600):
    n = xnew.size
    m_a_l = [[] for _ in range(model.num_sources)]
    v_a_l = [[] for _ in range(model.num_sources)]
    m_c_l = [[] for _ in range(model.num_sources)]
    v_c_l = [[] for _ in range(model.num_sources)]
    m_s_l = [[] for _ in range(model.num_sources)]

    for i in range(n/ws):
        x = xnew[i*ws : (i+1)*ws].copy()
        m_a, v_a = model.predict_act(x)
        m_c, v_c = model.predict_com(x)

        for j in range(model.num_sources):
            m_a_l[j].append(m_a[j].copy())
            v_a_l[j].append(v_a[j].copy())
            m_c_l[j].append(m_c[j].copy())
            v_c_l[j].append(v_c[j].copy())
            m_s_l[j].append(gpitch.logistic(m_a[j]) * m_c[j])

    for j in range(model.num_sources):
        m_a_l[j] = np.asarray(m_a_l[j]).reshape(-1, 1)
        v_a_l[j] = np.asarray(v_a_l[j]).reshape(-1, 1)
        m_c_l[j] = np.asarray(m_c_l[j]).reshape(-1, 1)
        v_c_l[j] = np.asarray(v_c_l[j]).reshape(-1, 1)
        m_s_l[j] = gpitch.logistic(m_a_l[j]) * m_c_l[j]

    return m_a_l, v_a_l, m_c_l, v_c_l, m_s_l



class Pdgp(gpflow.model.Model):
    def __init__(self, x, y, z, kern, whiten=True, minibatch_size=None, nlinfun=logistic_tf):
        """
        Pitch detection using Gaussian process.

        Constructor.
        :param x:
        :param y:
        :param z:
        :param kern_com:
        :param kern_act:
        :param transform:
        :param whiten:
        :param minibatch_size:
        """

        gpflow.model.Model.__init__(self)

        if minibatch_size is None:
            minibatch_size = x.shape[0]

        self.minibatch_size = minibatch_size
        self.num_data = x.shape[0]
        self.num_sources = len(kern[0])
        self.whiten = whiten
        self.nlinfun = nlinfun
        self.likelihood = MpdLik(nlinfun=self.nlinfun, num_sources=self.num_sources)

        self.x = MinibatchData(x, minibatch_size, np.random.RandomState(0))
        self.y = MinibatchData(y, minibatch_size, np.random.RandomState(0))

        self.kern_act = ParamList(kern[0])
        self.kern_com = ParamList(kern[1])

        self.num_inducing_c = []
        self.num_inducing_a = []

        za_l = []
        zc_l = []
        q_mu_com_l = []
        q_mu_act_l = []
        q_sqrt_com_l = []
        q_sqrt_act_l = []

        for i in range(self.num_sources):
            self.num_inducing_a.append(z[0][i].size)
            self.num_inducing_c.append(z[1][i].size)

            za_l.append(Param(z[0][i].copy() ))
            zc_l.append(Param(z[1][i].copy() ))

            q_mu_act_l.append(Param(np.zeros(z[0][i].shape)))
            q_mu_com_l.append(Param(np.zeros(z[1][i].shape)))

            q_sqrt_act_l.append(Param(np.array([np.eye(self.num_inducing_a[i]) for _ in range(1)]).swapaxes(0, 2)))
            q_sqrt_com_l.append(Param(np.array([np.eye(self.num_inducing_c[i]) for _ in range(1)]).swapaxes(0, 2)))


        self.za = ParamList(za_l)
        self.zc = ParamList(zc_l)
        self.q_mu_com = ParamList(q_mu_com_l)
        self.q_mu_act = ParamList(q_mu_act_l)
        self.q_sqrt_com = ParamList(q_sqrt_com_l)
        self.q_sqrt_act = ParamList(q_sqrt_act_l)

    def build_prior_kl(self):
        """
        compute KL divergences.
        :return:
        """

        if self.whiten:
            kl_act = [gauss_kl(self.q_mu_act[i], self.q_sqrt_act[i]) for i in range(self.num_sources)]
            kl_com = [gauss_kl(self.q_mu_com[i], self.q_sqrt_com[i]) for i in range(self.num_sources)]
        else:
            k_act, k_com = [], []
            kl_act, kl_com = [], []
            for i in range(self.num_sources):
                k_act.append(self.kern_act[i].K(self.za[i]) + tf.eye(self.num_inducing_a[i], dtype=float_type)*jitter)
                k_com.append(self.kern_com[i].K(self.zc[i]) + tf.eye(self.num_inducing_c[i], dtype=float_type)*jitter)
                kl_act.append(gauss_kl(self.q_mu_act[i], self.q_sqrt_act[i], k_act[i]))
                kl_com.append(gauss_kl(self.q_mu_com[i], self.q_sqrt_com[i], k_com[i]))

        return tf.reduce_sum(kl_act) + tf.reduce_sum(kl_com)

    def build_likelihood(self):
        """
        Compute the objective function
        :return:
        """
        kl = self.build_prior_kl()  # Get prior kl.

        # Get conditionals
        mean_act = self.num_sources*[None]
        mean_com = self.num_sources*[None]
        var_act = self.num_sources*[None]
        var_com = self.num_sources*[None]

        for i in range(self.num_sources):
            mean_act[i], var_act[i] = gpflow.conditionals.conditional(self.x, self.za[i],
                                                                      self.kern_act[i], self.q_mu_act[i],
                                                                      q_sqrt=self.q_sqrt_act[i],
                                                                      full_cov=False,  whiten=self.whiten)

            mean_com[i], var_com[i] = gpflow.conditionals.conditional(self.x, self.zc[i],
                                                                      self.kern_com[i], self.q_mu_com[i],
                                                                      q_sqrt=self.q_sqrt_com[i],
                                                                      full_cov=False, whiten=self.whiten)

        mean_act_concat = tf.concat(mean_act, 1)
        var_act_concat = tf.concat(var_act, 1)

        mean_com_concat = tf.concat(mean_com, 1)
        var_com_concat = tf.concat(var_com, 1)

        fmean = tf.concat([mean_act_concat, mean_com_concat], 1)
        fvar = tf.concat([var_act_concat, var_com_concat], 1)

        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.y)  # Get variational expectations

        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
            tf.cast(tf.shape(self.x)[0], settings.dtypes.float_type)  # re-scale for minibatch size
        return tf.reduce_sum(var_exp) * scale - kl

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_act(self, xnew):
        mean, var = self.num_sources*[None], self.num_sources*[None]
        for i in range(self.num_sources):
            mean[i], var[i] = gpflow.conditionals.conditional(xnew, self.za[i], self.kern_act[i],
                                                              self.q_mu_act[i], q_sqrt=self.q_sqrt_act[i],
                                                              full_cov=False, whiten=self.whiten)
        return mean, var

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_com(self, xnew):
        mean, var = self.num_sources*[None], self.num_sources*[None]
        for i in range(self.num_sources):
            mean[i], var[i] = gpflow.conditionals.conditional(xnew, self.zc[i], self.kern_com[i],
                                                              self.q_mu_com[i], q_sqrt=self.q_sqrt_com[i],
                                                              full_cov=False, whiten=self.whiten)
        return mean, var

    @gpflow.param.AutoFlow((tf.float64, [None, None]))
    def predict_act_n_com(self, xnew):

        mean_a, var_a = self.num_sources*[None], self.num_sources*[None]
        mean_c, var_c = self.num_sources*[None], self.num_sources*[None]
        mean_source = self.num_sources*[None]


        for i in range(self.num_sources):
            mean_a[i], var_a[i] = gpflow.conditionals.conditional(xnew, self.za[i], self.kern_act[i],
                                                                  self.q_mu_act[i], q_sqrt=self.q_sqrt_act[i],
                                                                  full_cov=False, whiten=self.whiten)

            mean_c[i], var_c[i] = gpflow.conditionals.conditional(xnew, self.zc[i], self.kern_com[i],
                                                                  self.q_mu_com[i], q_sqrt=self.q_sqrt_com[i],
                                                                  full_cov=False, whiten=self.whiten)

            mean_source[i] = self.nlinfun(mean_a[i])*mean_c[i]
        return mean_a, var_a, mean_c, var_c, mean_source
    
    


















#