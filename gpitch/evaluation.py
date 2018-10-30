import numpy as np
import time
import pickle
import soundfile
import tensorflow as tf
import gpitch


def evaluation(gpu='0', intrument_idx=0, window_size=14*16000, nivps=20, nfpc=1, minibatch_size=100,
               maxiter=1, use_centers=True, visualize_results=True, save=False):

    # initialize settings
    sess = gpitch.init_settings(gpu)  # select gpu
    list_inst = ['011PFNOM', '131EGLPM', '311CLNOM', 'ALVARADO']  # list of instruments
    inst = list_inst[intrument_idx]

    print("Analysing file {}, window size {}, GPU {}.".format(inst, window_size, 0))
    print("Iterations {}, number of features per frequency center {}, minibatch size {}".format(maxiter,
          nfpc, minibatch_size))

    # load test data
    testdata_directory = '/import/c4dm-04/alvarado/datasets/ss_amt/test_data/'
    test_file = inst + "_mixture.wav"
    x, y, fs = gpitch.readaudio(testdata_directory + test_file, frames=14*16000)
    num_windows = y.size / window_size
    xtest, ytest = gpitch.segmented(x, y, window_size=window_size)

    print("Number of windows to analyze {}".format(num_windows))

    # load train data for getting kernel features
    traindata_directoy = '/import/c4dm-04/alvarado/datasets/ss_amt/training_data/'
    train_files = gpitch.lfiles_training[intrument_idx]
    num_pitches = len(train_files)
    if0 = gpitch.find_ideal_f0(train_files)  # ideal frequency for each pitch

    # init lists to save features
    aux_list = num_pitches*[None]
    xtrain = list(aux_list)
    ytrain = list(aux_list)
    f_center = list(aux_list)
    v_center = list(aux_list)
    F = list(aux_list)
    S = list(aux_list)
    th = list(aux_list)
    frequency = list(aux_list)
    energy = list(aux_list)
    maxh = 25 
    totalnumf = maxh * nfpc 

    for i in range(num_pitches):
        # load train data
        xtrain[i], ytrain[i], fs = gpitch.readaudio(traindata_directoy + train_files[i])

        # get kernel features
        f_center[i], v_center[i], F[i], S[i], th[i] = gpitch.init_cparam(y=ytrain[i], fs=fs,
                                                                         maxh=maxh, ideal_f0=if0[i],
                                                                         scaled=False)
        frequency[i], energy[i] = gpitch.get_features(f=F[i], s=S[i], f_centers=f_center[i],
                                                      nfpc=nfpc, use_centers=use_centers, totalnumf=totalnumf)

    # initialization of models
    save_location = "/import/c4dm-04/alvarado/results/ss_amt/evaluation/new_kernel/models/"
    z, m, kern = [], [], []

    for i in range(num_windows):
        # init kernel
        kern.append(gpitch.init_kern(num_pitches=num_pitches, energy=energy, frequency=frequency))

        # init inducing variables
        z.append(gpitch.init_iv(x=xtest[i].copy(), num_sources=num_pitches, nivps_a=nivps,
                 nivps_c=nivps, fs=fs))

        # init model
        m.append(gpitch.pdgp.Pdgp(x=xtest[i], y=ytest[i], z=z[i], kern=kern[i],
                                  minibatch_size=minibatch_size))
        m[i].za.fixed = False
        m[i].zc.fixed = True

    # optimization
    results = []
    for i in range(num_windows):
        start_time = time.time()
        m[i].optimize(method=tf.train.AdamOptimizer(learning_rate=0.002, epsilon=0.1),
                      maxiter=maxiter)

        # compute prediction
        results.append(gpitch.pdgp.predict_windowed(model=m[i], xnew=xtest[i], ws=1600))
        m[i].save_prediction = list(results[i])
        print("Time optimizing and predicting {} secs".format(time.time() - start_time))

        # save models
        if num_windows == 1:
            pickle.dump(m[i], open(save_location + inst + "_full_window_nfpc_" + str(nfpc) + ".p",
                        "wb"))
        else:
            pickle.dump(m[i], open(save_location + inst + "_window_" + str(i+1) + "_nfpc_" +
                        str(nfpc) + ".p", "wb"))

        # reset tensorflow graph
        tf.reset_default_graph()

    # merge results
    results_merged = gpitch.window_overlap.merge_all(results)
    x_final, y_final, r_final = gpitch.window_overlap.get_results_arrays_noov(x=xtest, y=ytest,
                                                                              results=results_merged,
                                                                              window_size=window_size)

    # save wav files
    pitch_name = ['C', 'E', 'G']
    if save:
        for i in range(3):
            if num_windows == 1:
                name = inst + "_part_" + pitch_name[i] + "_full_window_nfpc_" + str(nfpc) + ".wav"
            else:
                name = inst + "_part_" + pitch_name[i] + "_windowed_nfpc_" + str(nfpc) + ".wav"
            print name
            aux = r_final[-1][i]/np.max(np.abs(r_final[-1][i]))
            soundfile.write(save_location + name, aux, fs)

    # visualize results
    if visualize_results:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (16, 3)
        import gpitch.myplots as mplt

        # plot spectral respresentation training data and selected features
        plt.figure(figsize=(16, 9))
        for i in range(num_pitches):
            plt.subplot(3, 1, i+1)
            plt.plot(F[i], S[i]/np.max(S[i]), 'xC0')
            plt.plot(frequency[i], energy[i]/np.max(energy[i]), 'sC1')
            plt.plot(f_center[i], v_center[i]/np.max(v_center[i]), 'vC2')
            plt.legend(["Spetral density data", "Features selected", "Frequency centers"])

        # plot prediction components and activations
        plt.figure(figsize=(16, 9))
        for i in range(num_windows):
            m_a, v_a, m_c, v_c, esource = m[i].save_prediction
            for j in range(num_pitches):
                plt.subplot(3, 2, 2*(j+1) - 1)
                mplt.plot_predict(xtest[i], m_a[j], v_a[j], m[i].za[j].value, plot_z=True,
                                  latent=True, plot_latent=False)

                plt.subplot(3, 2, 2*(j+1))
                mplt.plot_predict(xtest[i], m_c[j], v_c[j], m[i].zc[j].value, plot_z=False)

        # plot sources
        plt.figure(figsize=(16, 9))
        gpitch.window_overlap.plot_sources(x_final, y_final, r_final[-1])
    sess.close()
    return m
