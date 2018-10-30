import os
import numpy as np
import gpitch
from gpitch.methods import readaudio


def load_filenames(path, inst, names):
    files = np.asarray(os.listdir(path))  # load name all files in directory
    nfiles = len(files)  # number of files in directory

    flag = np.asarray(nfiles * [None])  # flag to mark if name of instrument present in filename
    for i in range(nfiles):  # mark file names with instrument in it
        flag[i] = files[i].find(inst)

    idx = np.where(flag != -1)[0]  # choose only file names with instrument
    files = files[idx]

    final_list = len(files) * [None]
    for i in range(len(names)):
        flag = np.asarray(len(files) * [None])  # flag to mark if name of pitch/mixture present in filename
        for j in range(len(files)):
            flag[j] = files[j].find(names[i])
        idx = np.where(flag != -1)[0]  # choose only file name specific pitch/mixture
        final_list[i] = files[idx][0]

    return final_list


def load_traindata(path, inst, frames=-1, start=0):
    """Load test data, that is, mixture and sources."""
    names = ['_M60_', '_M64_', '_M67_']
    filenames = load_filenames(path=path, inst=inst, names=names)

    x, aux, fs = readaudio(fname=path + filenames[0], frames=frames, start=start)
    traindata = [readaudio(fname=path + filenames[i], frames=frames, start=start, scaled=True)[1]
                 for i in range(len(filenames))]
    return x, traindata, fs, filenames


def load_testdata(path, inst, frames=-1, start=0):
    """Load test data, that is, mixture and sources."""
    names = ['mixture', '_C_', '_E_', '_G_']
    filenames = load_filenames(path=path, inst=inst, names=names)

    x, y, fs = readaudio(fname=path + filenames[0], frames=frames, start=start)
    sources = [readaudio(fname=path + filenames[i], frames=frames, start=start)[1] for i in range(1, len(filenames))]
    mix = sum(sources)
    return x, mix, sources, fs, filenames


def get_kernel_features(filenames, ytrain, maxh, fs):
    num_pitches = len(filenames)
    if0 = gpitch.find_ideal_f0(filenames)  # ideal frequency for each pitch
    all = [gpitch.init_cparam(y=ytrain[i], fs=fs, maxh=maxh, ideal_f0=if0[i], scaled=False) for i in range(num_pitches)]
    freq_feat = num_pitches*[None]
    var_feat =  num_pitches*[None]
    for i in range(num_pitches):
        freq_feat[i] = all[i][0].copy()
        var_feat[i] =  all[i][1].copy()
    return all, freq_feat, var_feat
