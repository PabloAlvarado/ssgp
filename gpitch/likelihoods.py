import gpflow
import tensorflow as tf
import numpy as np
import itertools
from gpflow.param import transforms


def exp_value_closed_form(mean, var, b):
    return tf.sqrt(b / (var + b)) * tf.exp(-0.5*(mean**2) / (var + b))

def mvhermgauss(means, covs, H, D):
    """
    Return the evaluation locations, and weights for several multivariate
    Hermite-Gauss quadrature runs.
    :param means: NxD
    :param covs: NxDxD
    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations (H**DxNxD), weights (H**D)
    """
    N = tf.shape(means)[0]
    gh_x, gh_w = gpflow.likelihoods.hermgauss(H)
    xn = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    wn = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    cholXcov = tf.cholesky(covs)  # NxDxD
    X = 2.0 ** 0.5 * tf.batch_matmul(cholXcov, tf.tile(xn[None, :, :],
                                     (N, 1, 1)), adj_y=True) + \
                                    tf.expand_dims(means, 2)  # NxDxH**D
    Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, D))  # H**DxNxD
    return Xr, wn * np.pi ** (-D * 0.5)

# Pablo Alvarado implementation
def hermgauss1d(mean_g, var_g, H, nlinfun):
    #H = 20  # get eval points and weights
    gh_x, gh_w = gpflow.quadrature.hermgauss(H)
    gh_x = gh_x.reshape(1, -1)
    gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)

    shape = tf.shape(mean_g)  # get  output shape
    X = gh_x * tf.sqrt(2.*var_g) + mean_g  # transformed evaluation points
    #evaluations = 1. / (1. + tf.exp(-X))  # sigmoid function
    evaluations = nlinfun(X)
    E1 = tf.reshape(tf.matmul(evaluations, gh_w), shape)  # compute expect
    E2 = tf.reshape(tf.matmul(evaluations**2, gh_w), shape)
    return E1, E2

def log_lik_exp(Y, mean_g, var_g, mean_f, var_f, E1, E2, noise_var, K):
    A_l = K*[None]
    B_l = K*[None]
    C_l = []
    
    for i in range(K):
        A_l[i] = E1[i]*mean_f[i]
        B_l[i] = E2[i]*(var_f[i] + mean_f[i]**2)
        
    for i in range(K-1):
        for j in range(i+1, K):
            C_l.append(E1[i]*mean_f[i]*E1[j]*mean_f[j])
    
    A = tf.add_n(A_l)
    B = tf.add_n(B_l)
    if K == 1:
        C = 0.*mean_f[0]
    else:
        C = 2.*tf.add_n(C_l)
    
    var_exp = -0.5*( (1./noise_var)*(Y**2 - 2.*Y*A + B + C) + np.log(2.*np.pi) + tf.log(noise_var) )
    return var_exp

class LooLik(gpflow.likelihoods.Likelihood):
    '''Leave One Out likelihood'''
    def __init__(self, version):
        gpflow.likelihoods.Likelihood.__init__(self)
        self.variance = gpflow.param.Param(1., transforms.positive)
        self.version = version
    def logp(self, F, Y):
        f1, g1 = F[:, 0], F[:, 1]
        f2, g2 = F[:, 2], F[:, 3]
        y = Y[:, 0]
        sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
        sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
        mean = sigma_g1 * f1 + sigma_g2 * f2
        return gpflow.densities.gaussian(y, mean, self.variance).reshape(-1, 1)

    def variational_expectations(self, Fmu, Fvar, Y):
        old_version = self.version
        if old_version:
            H = 5 # number of Gauss-Hermite evaluation points. (reduced  to 5)
            D = 4  # Number of input dimensions (increased from 2 to 4)
            Xr, w = mvhermgauss(Fmu, tf.matrix_diag(Fvar), H, D)
            w = tf.reshape(w, [-1, 1])
            f1, g1 = Xr[:, 0], Xr[:, 1]
            f2, g2 = Xr[:, 2], Xr[:, 3]
            y = tf.tile(Y, [H**D, 1])[:, 0]
            sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
            sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
            mean =  sigma_g1 * f1 + sigma_g2 * f2
            evaluations = gpflow.densities.gaussian(y, mean, self.variance)
            evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w),
                                                                tf.shape(Fmu)[0]])))
            return tf.matmul(evaluations, w)

        else:
            # variational expectations function, Pablo Alvarado implementation
            mean_f1 = Fmu[:, 0]  # get mean and var of each q dist, and reshape
            mean_g1 = Fmu[:, 1]
            var_f1 = Fvar[:, 0]
            var_g1 = Fvar[:, 1]

            mean_f2 = Fmu[:, 2]  # get mean and var of each q dist, and reshape
            mean_g2 = Fmu[:, 3]
            var_f2 = Fvar[:, 2]
            var_g2 = Fvar[:, 3]

            mean_f1, mean_g1, var_f1, var_g1 = [tf.reshape(e, [-1, 1]) for e in
                                               (mean_f1, mean_g1, var_f1, var_g1)]

            mean_f2, mean_g2, var_f2, var_g2 = [tf.reshape(e, [-1, 1]) for e in
                                               (mean_f2, mean_g2, var_f2, var_g2)]
            H = 20
            # calculate required quadratures
            E1, E2 = hermgauss1d(mean_g1, var_g1, H)
            E3, E4 = hermgauss1d(mean_g2, var_g2, H)

            # compute log-lik expectations under variational distribution
            var_exp = -0.5*((1./self.variance)*(Y**2 -
                       2.*Y*(mean_f1*E1 + mean_f2*E3) +
                      (var_f1 + mean_f1**2)*E2 +
                      2.* mean_f1*E1 * mean_f2*E3 +
                      (var_f2 + mean_f2**2)*E4) +
                      np.log(2.*np.pi) +
                      tf.log(self.variance))
            return var_exp


class ModLik(gpflow.likelihoods.Likelihood):
    '''Modulated GP likelihood'''
    def __init__(self, transfunc):
        gpflow.likelihoods.Likelihood.__init__(self)
        self.variance = gpflow.param.Param(1., transforms.positive)
        self.transfunc = transfunc

    def logp(self, F, Y):
        f, g = F[:, 0], F[:, 1]
        y = Y[:, 0]
        #sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
        sigma_g = self.transfunc(g)
        mean = f * sigma_g
        return gpflow.densities.gaussian(y, mean, self.variance).reshape(-1, 1)

    # variational expectations function, Pablo Alvarado implementation
    def variational_expectations(self, Fmu, Fvar, Y):
        H = 20  # get eval points and weights
        gh_x, gh_w = gpflow.quadrature.hermgauss(H)
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)

        mean_f = Fmu[:, 0]  # get mean and var of each q distribution, and reshape
        mean_g = Fmu[:, 1]
        var_f = Fvar[:, 0]
        var_g = Fvar[:, 1]
        mean_f, mean_g, var_f, var_g = [tf.reshape(e, [-1, 1]) for e in (mean_f,
                                        mean_g, var_f, var_g)]
        shape = tf.shape(mean_g)  # get  output shape
        X = gh_x * tf.sqrt(2.*var_g) + mean_g  # transformed evaluation points
        #evaluations = tf.exp(X)  # sigmoid function
        #evaluations = 1. / (1. + tf.exp(-X))  # sigmoid function
        evaluations = self.transfunc(X)
        E1 = tf.reshape(tf.matmul(evaluations, gh_w), shape)  # compute expectations
        #E1 = 1. / (1. + tf.exp(-mean_g / tf.sqrt(1. + 3.1416*var_g/8.)))
        
        #E2 = E1**2 +  var_g * (tf.exp(-mean_g)/(1. + tf.exp(-mean_g))**2)**2
        E2 = tf.reshape(tf.matmul(evaluations**2, gh_w), shape)

        # compute log-lik expectations under variational distribution
        var_exp = -0.5*((1./self.variance)*(Y**2 - 2.*Y*mean_f*E1 +
                  (var_f + mean_f**2)*E2) + np.log(2.*np.pi) +
                  tf.log(self.variance))
        return var_exp

    # # variational expectations function, gpflow modulated_GP version
    # def variational_expectations(self, Fmu, Fvar, Y):
    #     H = 20
    #     D = 2
    #     Fvar_matrix_diag = tf.matrix_diag(Fvar)
    #     Xr, w = mvhermgauss(Fmu, Fvar_matrix_diag, H, D)
    #     w = tf.reshape(w, [-1, 1])
    #     f, g = Xr[:, 0], Xr[:, 1]
    #     y = tf.tile(Y, [H**D, 1])[:, 0]
    #     sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
    #     mean = f * sigma_g
    #     evaluations = gpflow.densities.gaussian(y, mean, self.variance)
    #     evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w),
    #                                tf.shape(Fmu)[0]])))
    #     n_var_exp = tf.matmul(evaluations, w)
    #     return n_var_exp


class SsLik(gpflow.likelihoods.Likelihood):
    '''Source separation likelihood'''
    def __init__(self, nlinfun, quad=True):
        gpflow.likelihoods.Likelihood.__init__(self)
        self.variance = gpflow.param.Param(1., transforms.positive)
        self.nlinfun = nlinfun
        self.quad = quad

    def logp(self, F, Y):
        f1, g1 = F[:, 0], F[:, 1]
        f2, g2 = F[:, 2], F[:, 3]
        f3, g3 = F[:, 4], F[:, 5]
        y = Y[:, 0]
        #sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
        #sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
        #sigma_g3 = 1./(1 + tf.exp(-g3))  # squash g to be positive
        sigma_g1 = self.nlinfun(g1)  # squash g to be positive
        sigma_g2 = self.nlinfun(g2)  # squash g to be positive
        sigma_g3 = self.nlinfun(g3)  # squash g to be positive
        mean = sigma_g1 * f1 + sigma_g2 * f2 + sigma_g3 * f3
        return gpflow.densities.gaussian(y, mean, self.variance).reshape(-1, 1)

    def variational_expectations(self, Fmu, Fvar, Y):
        # variational expectations function, Pablo Alvarado implementation
        mean_f1 = Fmu[:, 0]  # get mean and var of each q dist, and reshape
        mean_g1 = Fmu[:, 1]
        var_f1 = Fvar[:, 0]
        var_g1 = Fvar[:, 1]

        mean_f2 = Fmu[:, 2]  # get mean and var of each q dist, and reshape
        mean_g2 = Fmu[:, 3]
        var_f2 = Fvar[:, 2]
        var_g2 = Fvar[:, 3]

        mean_f3 = Fmu[:, 4]  # get mean and var of each q dist, and reshape
        mean_g3 = Fmu[:, 5]
        var_f3 = Fvar[:, 4]
        var_g3 = Fvar[:, 5]

        mean_f1, mean_g1, var_f1, var_g1 = [tf.reshape(e, [-1, 1]) for e in
                                           (mean_f1, mean_g1, var_f1, var_g1)]

        mean_f2, mean_g2, var_f2, var_g2 = [tf.reshape(e, [-1, 1]) for e in
                                           (mean_f2, mean_g2, var_f2, var_g2)]

        mean_f3, mean_g3, var_f3, var_g3 = [tf.reshape(e, [-1, 1]) for e in
                                           (mean_f3, mean_g3, var_f3, var_g3)]
        
        # calculate required quadratures
        if self.quad:
            H = 20
            E1, E2 = hermgauss1d(mean_g1, var_g1, H, self.nlinfun)
            E3, E4 = hermgauss1d(mean_g2, var_g2, H, self.nlinfun)
            E5, E6 = hermgauss1d(mean_g3, var_g3, H, self.nlinfun)
        else:
            # compute expected values using closed form expressions
            E1 = exp_value_closed_form(mean=mean_g1, var=var_g1, b=0.5)
            E2 = exp_value_closed_form(mean=mean_g1, var=var_g1, b=0.25)
            
            E3 = exp_value_closed_form(mean=mean_g2, var=var_g2, b=0.5)
            E4 = exp_value_closed_form(mean=mean_g2, var=var_g2, b=0.25)
            
            E5 = exp_value_closed_form(mean=mean_g3, var=var_g3, b=0.5)
            E6 = exp_value_closed_form(mean=mean_g3, var=var_g3, b=0.25)
        

        # compute log-lik expectations under variational distribution
        var_exp = -0.5*((1./self.variance)*(Y**2 -
                   2.*Y*(mean_f1*E1 + mean_f2*E3 + mean_f3*E5) +
                  (var_f1 + mean_f1**2)*E2 +
                  (var_f2 + mean_f2**2)*E4 +
                  (var_f3 + mean_f3**2)*E6 +
                  2.* (mean_f1*E1 * mean_f2*E3 + mean_f1*E1 * mean_f3*E5 + mean_f2*E3 * mean_f3*E5)) +
                  np.log(2.*np.pi) +
                  tf.log(self.variance))
        return var_exp




class MpdLik(gpflow.likelihoods.Likelihood):
    '''Modulated GP likelihood'''
    def __init__(self, nlinfun, num_sources):
        gpflow.likelihoods.Likelihood.__init__(self)
        self.variance = gpflow.param.Param(1., transforms.positive)
        self.nlinfun = nlinfun
        self.num_sources = num_sources

    def logp(self, F, Y):
#         if self.num_sources == 1:
#             g, f = F[:, 0], F[:, 1] 
#             sigma_g = self.nlinfun(g)  # squash g to be positive
#             mean = f * sigma_g
        
#         elif self.num_sources == 2:
#             g1, g2 = F[:, 0], F[:, 1]
#             f1, f2 = F[:, 2], F[:, 3]
#             sigma_g1 = 1./(1 + tf.exp(-g1))  # squash g to be positive
#             sigma_g2 = 1./(1 + tf.exp(-g2))  # squash g to be positive
#             mean = sigma_g1 * f1 + sigma_g2 * f2
            
#         elif self.num_sources == 3:
#             f1, g1 = F[:, 3], F[:, 0]
#             f2, g2 = F[:, 4], F[:, 1]
#             f3, g3 = F[:, 5], F[:, 2]
#             sigma_g1 = self.nlinfun(g1)  # squash g to be positive
#             sigma_g2 = self.nlinfun(g2)  # squash g to be positive
#             sigma_g3 = self.nlinfun(g3)  # squash g to be positive
#             mean = sigma_g1 * f1 + sigma_g2 * f2 + sigma_g3 * f3
#         else:
        g_l = self.num_sources*[None]
        f_l = self.num_sources*[None]
        sigma_g_l = self.num_sources*[None]
        mean_l = self.num_sources*[None]
        
        for i in range(self.num_sources):
            g_l[i] = F[:, i]
            f_l[i] = F[:, i + self.num_sources] 
            sigma_g_l[i] = self.nlinfun(g_l[i])
            mean_l[i] = sigma_g_l[i] * f_l[i]

        mean = tf.add_n(mean_l)  
        y = Y[:, 0]
        return gpflow.densities.gaussian(y, mean, self.variance).reshape(-1, 1)

    # variational expectations function, Pablo Alvarado implementation
    def variational_expectations(self, Fmu, Fvar, Y):
#         if self.num_sources == 1:

#             mean_f = Fmu[:, 1]  # get mean and var of each q distribution, and reshape
#             mean_g = Fmu[:, 0]
            
#             var_f = Fvar[:, 1]
#             var_g = Fvar[:, 0]
            
#             mean_f, mean_g, var_f, var_g = [tf.reshape(e, [-1, 1]) for e in (mean_f,
#                                             mean_g, var_f, var_g)]
            
#             H = 20  # get eval points and weights
#             E1, E2 = hermgauss1d(mean_g, var_g, H, self.nlinfun)

#             # compute log-lik expectations under variational distribution
#             var_exp = -0.5*((1./self.variance)*(Y**2 - 2.*Y*mean_f*E1 +
#                       (var_f + mean_f**2)*E2) + np.log(2.*np.pi) +
#                       tf.log(self.variance))
        
#         elif self.num_sources == 2:
#             mean_g1 = Fmu[:, 0] # get mean and var of each q dist, and reshape
#             mean_g2 = Fmu[:, 1]
#             mean_f1 = Fmu[:, 2] 
#             mean_f2 = Fmu[:, 3] 
            
#             var_g1 = Fvar[:, 0]
#             var_g2 = Fvar[:, 1]
#             var_f1 = Fvar[:, 2]
#             var_f2 = Fvar[:, 3]

#             mean_f1, mean_g1, var_f1, var_g1 = [tf.reshape(e, [-1, 1]) for e in
#                                                (mean_f1, mean_g1, var_f1, var_g1)]

#             mean_f2, mean_g2, var_f2, var_g2 = [tf.reshape(e, [-1, 1]) for e in
#                                                (mean_f2, mean_g2, var_f2, var_g2)]
#             H = 20
#             # calculate required quadratures
#             E1, E2 = hermgauss1d(mean_g1, var_g1, H, self.nlinfun)
#             E3, E4 = hermgauss1d(mean_g2, var_g2, H, self.nlinfun)

#             # compute log-lik expectations under variational distribution
#             var_exp = -0.5*((1./self.variance)*(Y**2 -
#                        2.*Y*(mean_f1*E1 + mean_f2*E3) +
#                       (var_f1 + mean_f1**2)*E2 +
#                       2.* mean_f1*E1 * mean_f2*E3 +
#                       (var_f2 + mean_f2**2)*E4) +
#                       np.log(2.*np.pi) +
#                       tf.log(self.variance))
            
#         elif self.num_sources == 3:
#             # variational expectations function, Pablo Alvarado implementation
#             mean_g1 = Fmu[:, 0]
#             mean_g2 = Fmu[:, 1]
#             mean_g3 = Fmu[:, 2]

#             mean_f1 = Fmu[:, 3] 
#             mean_f2 = Fmu[:, 4]  
#             mean_f3 = Fmu[:, 5]  

#             var_g1 = Fvar[:, 0]
#             var_g2 = Fvar[:, 1]
#             var_g3 = Fvar[:, 2]
            
#             var_f1 = Fvar[:, 3]
#             var_f2 = Fvar[:, 4]            
#             var_f3 = Fvar[:, 5]


#             mean_f1, mean_g1, var_f1, var_g1 = [tf.reshape(e, [-1, 1]) for e in
#                                                (mean_f1, mean_g1, var_f1, var_g1)]

#             mean_f2, mean_g2, var_f2, var_g2 = [tf.reshape(e, [-1, 1]) for e in
#                                                (mean_f2, mean_g2, var_f2, var_g2)]

#             mean_f3, mean_g3, var_f3, var_g3 = [tf.reshape(e, [-1, 1]) for e in
#                                                (mean_f3, mean_g3, var_f3, var_g3)]

#             # calculate required quadratures
#             H = 20
#             E1, E2 = hermgauss1d(mean_g1, var_g1, H, self.nlinfun)
#             E3, E4 = hermgauss1d(mean_g2, var_g2, H, self.nlinfun)
#             E5, E6 = hermgauss1d(mean_g3, var_g3, H, self.nlinfun)



#             # compute log-lik expectations under variational distribution
#             var_exp = -0.5*((1./self.variance)*(Y**2 -
#                        2.*Y*(mean_f1*E1 + mean_f2*E3 + mean_f3*E5) +
#                       (var_f1 + mean_f1**2)*E2 +
#                       (var_f2 + mean_f2**2)*E4 +
#                       (var_f3 + mean_f3**2)*E6 +
#                       2.* (mean_f1*E1 * mean_f2*E3 + mean_f1*E1 * mean_f3*E5 + mean_f2*E3 * mean_f3*E5)) +
#                       np.log(2.*np.pi) +
#                       tf.log(self.variance))
#         else:

        init_l = self.num_sources*[None] 
        mean_g_l = list(init_l)
        mean_f_l = list(init_l)
        var_g_l = list(init_l)
        var_f_l = list(init_l)
        Evalue1 = list(init_l) 
        Evalue2 = list(init_l)
        H = 20
        for i in range(self.num_sources):
            mean_g_l[i] = Fmu[:, i]
            mean_f_l[i] = Fmu[:, i + self.num_sources]

            var_g_l[i] = Fvar[:, i]
            var_f_l[i] = Fvar[:, i + self.num_sources]

            mean_g_l[i] = tf.reshape(mean_g_l[i], [-1, 1])
            mean_f_l[i] = tf.reshape(mean_f_l[i], [-1, 1])

            var_g_l[i] = tf.reshape(var_g_l[i], [-1, 1])
            var_f_l[i] = tf.reshape(var_f_l[i], [-1, 1])

            Evalue1[i], Evalue2[i] = hermgauss1d(mean_g_l[i], var_g_l[i], H, self.nlinfun)

        var_exp = log_lik_exp(Y, mean_g_l, var_g_l, mean_f_l, var_f_l, Evalue1, Evalue2, self.variance, self.num_sources) 
 
        return var_exp

















#
