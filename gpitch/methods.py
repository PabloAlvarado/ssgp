import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import gpflow
from scipy.fftpack import fft
from scipy import signal
import os
import fnmatch
import tensorflow as tf
import peakutils
import soundfile
import pickle
import time


def loadm(directory, pattern=''):
    """load an already gpitch trained model"""
    filenames = []
    filenames += [i for i in os.listdir(directory) if pattern in i]
    m_list = []  # list of models loaded
    for i in range(len(filenames)):
        m_list.append(pickle.load(open(directory + filenames[i], "rb")))
    return m_list, filenames


def find_ideal_f0(string):
    """"""
    ideal_f0 = []
    for j in range(len(string)):
        for i in range(21, 109):
            if string[j].find('M' + str(i)) is not -1:
                ideal_f0.append(midi2freq(i))
    return ideal_f0


def readaudio(fname, frames=-1, start=0, aug=False, scaled=False):
    y, fs = soundfile.read(fname, frames=frames, start=start)  # load data and sample freq
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] == 2:  # convert to mono
        y = np.mean(y, 1) 
    y = y.reshape(-1, 1)
    if scaled:
        beta = np.max(np.abs(y))
        if beta == 0.:
            beta = 1.
        y /= beta
    if aug:
        augnum = 1000  # number of zeros to add
        y = np.append(np.zeros((augnum, 1)), y).reshape(-1, 1)
        #y = np.append(y, np.zeros((augnum, 1))).reshape(-1, 1)
    frames = y.size
    x = np.linspace(0., (frames-1.)/fs, frames).reshape(-1, 1)  # time vector
    return x, y, fs


def trim_n_merge(x, trim_size=1600, aug=True):
    xl = []
    for i in range(len(x)):
        if aug:
            xl.append(x[i][trim_size:-trim_size].copy().reshape(-1, 1))
        else:
            xl.append(x[i].copy().reshape(-1, 1))
    xl = np.asarray(xl).reshape(-1, 1)
    return xl

def merge_all(inlist):
    outlist = [[[], [], []],
               [[], [], []],
               [[], [], []],
               [[], [], []],
               [],
               []]
    for j in range(4):
        for i in range(2):
            outlist[j][0].append(inlist[j][i][0])
            outlist[j][1].append(inlist[j][i][1])
            outlist[j][2].append(inlist[j][i][2])
    outlist[4] = inlist[4]
    outlist[5] = inlist[5]

    for j in range(4):
        outlist[j][0] = trim_n_merge(outlist[j][0])
        outlist[j][1] = trim_n_merge(outlist[j][1])
        outlist[j][2] = trim_n_merge(outlist[j][2])
    outlist[4] =  trim_n_merge(outlist[4])
    outlist[5] =  trim_n_merge(outlist[5])

    return outlist

def init_cparam(y, fs, maxh, ideal_f0, scaled=True, win_size=10, thres=0.1, min_dis=0.8):
    '''
    :param y: data
    :param fs: sample frequency
    :param maxh: max number of partials
    :param ideal_f0: ideal f0 or pitch
    :param scaled: to scale or not the variance
    :param win_size: size of window to smooth spectrum
    :return:
    '''

    N = y.size
    Y = fft(y.reshape(-1,)) #  FFT data
    S =  2./N * np.abs(Y[0:N//2]) #  spectral density data
    F = np.linspace(0, fs/2., N//2) #  frequency vector

    win =  signal.hann(win_size)
    Ss = signal.convolve(S, win, mode='same') / sum(win)

    Sslog = np.log(S)
    Sslog = Sslog + np.abs(np.min(Sslog))
    Sslog /= np.max(Sslog)
    thres = thres*np.max(Sslog)
    min_dist = min_dis*np.argmin(np.abs(F - ideal_f0))
    idx = peakutils.indexes(Sslog, thres=thres, min_dist=min_dist)

    F_star, S_star = F[idx], S[idx]

    idx_sorted = np.argsort(F_star.copy())
    S_star = S_star[idx_sorted]
    F_star = np.sort(F_star)


    for index in range(F_star.size):
        if F_star[index] < 0.75*ideal_f0:
            F_star2 = np.delete(F_star, [index])
            S_star2 = np.delete(S_star, [index])
        else:
            F_star2 = F_star.copy()
            S_star2 = S_star.copy()

    aux1 = np.flip(np.sort(S_star2), 0)
    aux2 = np.flip(np.argsort(S_star2), 0)

    if aux1.size > maxh :
        vvec = aux1[0:maxh]
        idxf = aux2[0:maxh]
    else :
        vvec = aux1
        idxf = aux2

    if scaled:
        sig_scale = 1./ np.sum(vvec) #rescale (sigma)
        vvec *= sig_scale

    freq_final = F_star2[idxf]
    var_final = vvec

    idx_sorted = np.argsort(freq_final.copy())
    var_final = var_final[idx_sorted]
    freq_final = np.sort(freq_final)

    return [freq_final, var_final, F, S, thres]

def init_settings(visible_device='0', interactive=False, allow_growth=True, run_on_server=True):
    '''
    Initialize usage of GPU and plotting visible_device : which GPU to use
    '''

    # deactivate tf warnings (default 0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # configuration use only one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    # configuration to not to use all the memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth

    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)

    # run on server or run on local machine
    if run_on_server:
        path = "/import/"
    else:
        path = "/"
    return sess, path


def load_filenames(directory, pattern, pitches=None, ext=".wav"):
    auxl = fnmatch.filter(os.listdir(directory),  '*' + pattern + '*' + ext)
    if pitches is not None:
        filel = [fnmatch.filter(auxl, '*_M' + str(pitch) + '_*')[0] for pitch in pitches]
    else:
        filel = auxl
    filel = np.asarray(filel).reshape(-1,)
    return filel


def norm(x):
    """divide by absolute max"""
    return x / np.max(np.abs(x))

def logistic(x):
    """ logistic function """
    return 1./(1. + np.exp( -2.*(x - np.pi) ) )

def ilogistic(x):
    """inverse logistic function"""
    return - np.log(1./x - 1.)

def softplus(x):
    """ softplus function """
    return np.log(np.exp(x) + 1.)

def isoftplus(x):
    """ inverse softplus function """
    return np.log(np.exp(x) - 1.)

def gaussfun(x):
    return np.exp(-2.*(x - np.pi)**2)

def logistic_tf(x):
    """logistic function using tensorflow """
    return 1./(1. + tf.exp(-2.*(x - np.pi) ))

def softplus_tf(x):
    """ softplus function using tensorflow  """
    return tf.log(tf.exp(x) + 1.)

def isoftplus_tf(x):
    """ inverse softplus function using tensorflow  """
    return tf.log(tf.exp(x) - 1.)

def ilogistic_tf(x):
    """inverse logistic function using tensorflow"""
    return - tf.log(1./x - 1.)

def gaussfun_tf(x):
    return tf.exp(-2.*(x - np.pi)**2)


def load_pitch_params_data(pitch_list, data_loc, params_loc):
    '''
    This function loads the desired pitches and the gets the names of the files in the MAPS dataset
    corresponding to those pitches. Also returns the learned params and data related to
    those files.
    '''
    intensity = 'F'  # property maps datset, choose "forte" sounds
    Np = pitch_list.size # number of pitches
    filename_list =[None]*Np
    lfiles = load_filename_list(data_loc + 'filename_list.txt')
    j = 0
    for pitch in pitch_list:
        for i in lfiles:
            if pitch in i:
                if intensity in i:
                    filename_list[j] = i
                    j += 1
    final_list  = np.asarray(filename_list).reshape(-1, )
    train_data = [None]*Np #  load training data and learned params
    params = [None]*Np
    for i in range(Np):
        N = 32000 # numer of data points to load
        fs, aux = wavread(data_loc + final_list[i] + '.wav', start=5000, N=N)
        train_data[i] = aux.copy()
        x = np.linspace(0, (N-1.)/fs, N).reshape(-1, 1)
        params[i] = np.load(params_loc + 'params_act_' + final_list[i] + '.npz')
        keys = np.asarray(params[i].keys()).reshape(-1,)
    return final_list, train_data, params


def midi2freq(midi):
    return 2.**( (midi - 69.)/12. ) * 440.

def freq2midi(freq):
    return int(69. + 12. * np.log2(freq / 440.))


lfiles_training = [ ['011PFNOM_M60_train.wav',
                    '011PFNOM_M64_train.wav',
                    '011PFNOM_M67_train.wav'],
                    ['131EGLPM_M60_train.wav',
                    '131EGLPM_M64_train.wav',
                    '131EGLPM_M67_train.wav'],
                    ['311CLNOM_M60_train.wav',
                    '311CLNOM_M64_train.wav',
                    '311CLNOM_M67_train.wav'],
                    ['ALVARADO_M60_train.wav',
                    'ALVARADO_M64_train.wav',
                    'ALVARADO_M67_train.wav']]

























#
