import tensorflow as tf
import numpy as np
import gpitch
import time
import pickle
import gpitch.myplots as mplt


def train_notebook(gpu='0', list_limits=None, maxiter=[1000, 10000], nivps=[200, 200], frames=8000, save=True):

    sess = gpitch.init_settings(gpu)  # choose gpu to work

    ## import 12 audio files for intializing component parameters
    datadir = '/import/c4dm-04/alvarado/datasets/ss_amt/training_data/'
    lfiles = gpitch.lfiles_training
    lfiles = lfiles[list_limits[0]:list_limits[1]]
    numf = len(lfiles)  # number of files loaded
    if0 = gpitch.find_ideal_f0(lfiles)  # ideal frequency for each pitch
    x2, y2, fs2 = [], [], []
    for i in range(numf):
        a, b, c = gpitch.readaudio(datadir + lfiles[i], frames=32000, aug=False)
        x2.append(a.copy())
        y2.append(b.copy())
        fs2.append(c)
    lkernel, iparam = gpitch.init_models.init_kernel_training(y=y2, list_files=lfiles)

    ## Compare FFT kernels and initialization data
    array0 = np.asarray(0.).reshape(-1,1)
    x_p = np.linspace(-5, 5, 10*16000).reshape(-1, 1)
    k_p = []
    for i in range(numf):
        k_p.append(lkernel[1][i].compute_K(x_p, array0))
    Fdata = np.linspace(0., 8000., 16000).reshape(-1, 1)
    Fkernel = np.linspace(0., 8000., 5*16000).reshape(-1, 1)
    mplt.plot_fft(Fdata, Fkernel, y2, k_p, numf, iparam)

    ## import 12 audio files for training (same data but only 0.5 seconds)
    n = frames
    x, y, fs = [], [], []
    for i in range(numf):
        a, b, c = gpitch.readaudio(datadir + lfiles[i], frames=n, aug=True)
        x.append(a.copy())
        y.append(b.copy())
        fs.append(c)

    ## initialize models
    m = []
    nivps_a, nivps_c = nivps[0], nivps[1]  # num inducing variables per second for act and comp
    nlinfun = gpitch.logistic
    for i in range(numf):
        z = gpitch.init_iv(x=x[i], num_sources=numf, nivps_a=nivps_a, nivps_c=nivps_c, fs=fs[i])
        kern = [ [lkernel[0][i]], [lkernel[1][i]] ]
        m.append(gpitch.pdgp.Pdgp(x=x[i], y=y[i], z=z, kern=kern))
        m[i].za.fixed = True
        m[i].zc.fixed = True

    ## optimization
    for i in range(numf):
        st = time.time()
        #m[i].kern_act[0].variance.fixed = True
        #m[i].kern_com[0].lengthscales.fixed = True
        m[i].optimize(disp=1, maxiter=maxiter[0])
        m[i].za.fixed = False
        m[i].optimize(disp=1, maxiter=maxiter[1])
        print("model {}, time optimizing {} sec".format(i+1, time.time() - st))
        tf.reset_default_graph()

    ## prediction
    m_a, v_a = [], []  # list mean, var activation
    m_c, v_c = [], []  # list mean, var component
    m_s = []  # mean source
    for i in range(numf):
        st = time.time()
        mean_act, var_act = m[i].predict_act(x[i])
        mean_com, var_com = m[i].predict_com(x[i])
        print("model {}, time predicting {}".format(str(i + 1), time.time() - st) )
        m_s.append(gpitch.logistic(mean_act[0])*mean_com[0])
        m_a.append(mean_act[0])
        m_c.append(mean_com[0])
        v_a.append(var_act[0])
        v_c.append(var_com[0])
        tf.reset_default_graph()

    ## plots
    for i in range(len(m_s)):
        mplt.plot_training_all(x=x[i], y=y[i], source=m_s[i], m_a=m_a[i], v_a=v_a[i], m_c=m_c[i], v_c=v_c[i], m=m[i],
                             nlinfun=nlinfun)
    mplt.plot_parameters(m)
    # k_p2 = []
    # for i in range(numf):
    #     k_p2.append(m[i].kern_com[0].compute_K(x_p, array0))
    # gpitch.pltrain.plot_fft(Fdata, Fkernel, y2, k_p2, numf, iparam)

    ## save models
    if save:
        for i in range(numf):
            m[i].prediction_act = [m_a[i], v_a[i]]
            m[i].prediction_com = [m_c[i], v_c[i]]
            location = "/import/c4dm-04/alvarado/results/ss_amt/train/logistic/trained_" +  lfiles[i].strip('.wav')+".p"
            pickle.dump(m[i], open(location, "wb"))

    return m
