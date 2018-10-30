import tensorflow as tf
import numpy as np
import gpitch
import os
import time
import soundfile
import pickle
import matplotlib.pyplot as plt
from gpitch import window_overlap, logistic
import gpitch.myplots as mplt


plt.rcParams['figure.figsize'] = (16, 3)  # set plot size

di = 3*[None]
di[0] = "/import/c4dm-04/alvarado/results/ss_amt/train/logistic/"  # location saved models
di[1] = "/import/c4dm-04/alvarado/datasets/ss_amt/test_data/" # location test data
di[2] = "/import/c4dm-04/alvarado/results/ss_amt/evaluation/logistic/"  #location save results

def predict_windowed(x, pred_fun, ws=8000):
    n = x.size
    #results_w = []
    m_a = [[], [], []]
    v_a = [[], [], []]
    m_c = [[], [], []]
    v_c = [[], [], []]
    s_l = [[], [], []]
    for i in range(n/ws):
        xnew = x[i*ws : (i+1)*ws].copy()
        results_w = pred_fun(xnew)

        for j in range(3):
            m_a[j].append(results_w[0][j].copy())
            v_a[j].append(results_w[1][j].copy())
            m_c[j].append(results_w[2][j].copy())
            v_c[j].append(results_w[3][j].copy())
            s_l[j].append(results_w[4][j].copy())

    for j in range(3):
        m_a[j] = np.asarray(m_a[j]).reshape(-1, 1)
        v_a[j] = np.asarray(v_a[j]).reshape(-1, 1)
        m_c[j] = np.asarray(m_c[j]).reshape(-1, 1)
        v_c[j] = np.asarray(v_c[j]).reshape(-1, 1)
        s_l[j] = np.asarray(s_l[j]).reshape(-1, 1)
    return m_a, v_a, m_c, v_c, s_l


def evaluation_notebook(gpu='0',
                        inst=0,
                        nivps=[20, 20],
                        maxiter=[500, 500],
                        frames=16000,
                        window_size=16000,
                        minibatch_size=-1,
                        learning_rate=[0.0025, 0.0025],
                        opt_za=True,
                        windowed=False,
                        start=0,
                        save=False,
                        filename=None,
                        directory=di):
    """
    param nivps: number of inducing variables per second, for activations and components
    """

    ## settings
    if save:
        print("Results are going to be saved")
    else:
        print("Results are NOT going to be saved")

    if minibatch_size == -1:
        minibatch_size = int(np.sqrt(window_size))


    window_size_predic = 4000

    if frames < window_size:
        window_size = frames

    if frames < window_size_predic:
        window_size_predic = frames

    #if minibatch_size == None:
    #    window_size_predic = window_size




    sess = gpitch.init_settings(gpu)  # select gpu to use
    nlinfun = gpitch.logistic_tf  # use logistic or gaussian

    ## load pitch models
    linst = ['011PFNOM', '131EGLPM', '311CLNOM', 'ALVARADO']  # list of instruments
    instrument = linst[inst]
    pattern = instrument  # which model version
    m, names_list = gpitch.loadm(directory=directory[0], pattern=pattern)
    mplt.plot_trained_models(m, instrument)

    ## load test data
    test_data_dir = directory[1]
    lfiles = []
    lfiles += [i for i in os.listdir(test_data_dir) if instrument + '_mixture' in i]
    xall, yall, fs = gpitch.readaudio(test_data_dir + lfiles[0], aug=False, start=start, frames=frames)

    if windowed:
        #yall2 = np.vstack((  yall.copy(), 0.  ))
        #xall2 = np.vstack((  xall.copy(), xall[-1].copy() + xall[1].copy()  ))
        #x, y = window_overlap.windowed(xall2.copy(), yall2.copy(), ws=window_size)  # return list of segments
        x, y = window_overlap.segment(xall, yall, window_size=window_size, aug=False)
    else:
        x, y = [xall.copy()], [yall.copy()]
    results_list = len(x)*[None]
    var_params_list = [[], [], [], [], []]
    z_location_list = len(x)*[None]

    ## analyze whole signal by windows
    for i in range(len(y)):

        if i == 0:
            ## initialize model (do this only once)
            z = gpitch.init_iv(x=x[i], num_sources=3, nivps_a=nivps[0], nivps_c=nivps[1], fs=fs)  # location inducing var
            kern = gpitch.init_kernel_with_trained_models(m, option_two=False)
            mpd = gpitch.pdgp.Pdgp(x[i].copy(), y[i].copy(), z, kern, minibatch_size=minibatch_size, nlinfun=nlinfun)
            mpd.za.fixed = False
            mpd.zc.fixed = True
        else:
             ## reset model to analyze a new window
            gpitch.reset_model(m=mpd, x=x[i].copy(), y=y[i].copy(), nivps=nivps, m_trained=m, option_two=False)

        ## plot training data (windowed)
        plt.figure(5, figsize=(16,3)), plt.title("Test data  " + lfiles[0])
        plt.plot(mpd.x.value, mpd.y.value)

        ## optimization
        st = time.time()
        if minibatch_size is None:
            print ("VI optimization")
            mpd.optimize(disp=True, maxiter=maxiter[0])
        else:
            print ("SVI optimization")
            mpd.optimize(method=tf.train.AdamOptimizer(learning_rate=learning_rate[0], epsilon=0.1), maxiter=maxiter[0])

        ## optimization location inducing variables
        if opt_za:
            mpd.za.fixed = False
            st = time.time()
            if minibatch_size is None:
                print ("VI optimizing location inducing variables")
                mpd.optimize(disp=True, maxiter=maxiter[1])
            else:
                print ("SVI optimizing location inducing variables")
                mpd.optimize(method=tf.train.AdamOptimizer(learning_rate=learning_rate[1], epsilon=0.1), maxiter=maxiter[1])
            mpd.za.fixed = True
        print("Time optimizing {} secs".format(time.time() - st))

        ## prediction
        if 1:
            st = time.time()
            results_list[i] = predict_windowed(x=x[i].copy(), pred_fun=mpd.predict_act_n_com, ws=window_size_predic)
            print("Time predicting {} secs".format(time.time() - st))
        else:
            st = time.time()
            results_list[i] = mpd.predict_act_n_com(x[i].copy())
            print("Time predicting {} secs".format(time.time() - st))

        ## plot partial results
        plt.figure(123456789, figsize=(16, 4*6))
        mplt.plot_pdgp(x=x[i], y=y[i], m=mpd, list_predictions=results_list[i])

        ## save partial results
        var_params_list[0].append(mpd.q_mu_act)
        var_params_list[1].append(mpd.q_sqrt_act)
        var_params_list[2].append(mpd.q_mu_com)
        var_params_list[3].append(mpd.q_sqrt_com)
        var_params_list[4].append(mpd.likelihood.variance)
        z_location_list[i] = list(z)

        mpd.save_prediction = results_list[i]
        if windowed:
            pickle.dump(mpd, open("/import/c4dm-04/alvarado/results/ss_amt/evaluation/logistic/models/" + instrument +
                                  "_window_" + str(i+1) + ".p", "wb"))
        else:
            pickle.dump(mpd, open("/import/c4dm-04/alvarado/results/ss_amt/evaluation/logistic/models/" + instrument +
                                  "_full_window.p", "wb"))

        ## reset tensorflow graph
        tf.reset_default_graph()

    ## merge and overlap prediction results
    if windowed:
        rl_merged = window_overlap.merge_all(results_list)  # results merged
        x_final, y_final, r_final = window_overlap.get_results_arrays_noov(x=x, y=y, results=rl_merged, window_size=window_size)
        s_final = list(r_final[-1])
    else:
        x_final, y_final, s_final = x[0].copy(), y[0].copy(), results_list[0][4]

    ## plot sources
    plt.figure(figsize=(16, 9))
    window_overlap.plot_sources(x_final, y_final, s_final)

    final_results = [x_final, y_final, s_final]
    ## save wav files estimated sources
    if save:
        location_save = directory[2]
        for i in range(3):
            name = names_list[i].strip('_trained.p') + "_part" + filename + ".wav"
            soundfile.write(location_save + name, final_results[2][i]/np.max(np.abs(final_results[2][i])), 16000)

    ## group results
    all_results = [results_list, var_params_list, z_location_list]
    # return mpd, results, final_results

    return mpd, final_results, all_results



def logistic_20ivps_full_window(gpu, inst):
    return evaluation_notebook(gpu=gpu,
                               inst=inst,
                               nivps=[50, 100],
                               maxiter=[40000, 0],
                               frames=14*16000,
                               window_size=14*16000,
                               minibatch_size=-1,
                               opt_za=False,
                               windowed=False,
                               save=True,
                               filename="_logistic_20ivps_full_window",
                               directory=di)


def logistic_200ivps_windowed(gpu, inst):
    return evaluation_notebook(gpu=gpu,
                               inst=inst,
                               nivps=[200, 200],
                               maxiter=[10000, 0],
                               frames=14*16000,
                               window_size=32000,
                               minibatch_size=-1,
                               opt_za=False,
                               windowed=True,
                               save=True,
                               filename="_logistic_200ivps_windowed",
                               directory=di)






#
