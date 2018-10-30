from gpflow.kernels import Matern32, Matern12
from gpitch.kernels import MercerCosMix
import gpflow
import pickle


def init_kern_act(num_pitches):
    """Initialize kernels for activations and components"""

    kern_act = []

    for i in range(num_pitches):
        kern_act.append(Matern32(1, lengthscales=1.0, variance=3.5))
    return kern_act


def init_kern_com(num_pitches, lengthscale, energy, frequency, len_fixed=True):
    """Initialize kernels for activations and components"""

    kern_com, kern_exp, kern_per = [], [], []

    for i in range(num_pitches):
        kern_exp.append(Matern12(1, lengthscales=lengthscale[i].copy(), variance=1.0) )
        kern_exp[i].lengthscales.fixed = len_fixed

        kern_per.append(MercerCosMix(1, energy=energy[i].copy(), frequency=frequency[i].copy(), variance=1.0,))
        kern_per[i].fixed = True

        kern_com.append( kern_exp[i] * kern_per[i] )
    return kern_com


def init_kern(num_pitches, lengthscale, energy, frequency):
    """Initialize kernels for activations and components"""

    kern_act = init_kern_act(num_pitches)
    kern_com = init_kern_com(num_pitches, lengthscale, energy, frequency)
    kern = [kern_act, kern_com]
    return kern


def load_params(num_sources, fname):
    """ load kernel hyperparams for initialization"""
    path_p = '/import/c4dm-04/alvarado/results/sampling_covariance/'
    # path_p = '/home/pa/Desktop/sampling_covariance/'
    pitches = ["60", "64", "67"]
    hparam = []
    lengthscale = []
    variance = []
    frequency = []

    for i in range(num_sources):
        hparam.append(pickle.load(open(path_p + fname + "_M" + pitches[i] + "_hyperparams.p", "rb")))
        lengthscale.append(hparam[i][1].copy())
        variance.append(hparam[i][2].copy() / sum(hparam[i][2].copy()))
        frequency.append(hparam[i][3].copy())

    return lengthscale, variance, frequency