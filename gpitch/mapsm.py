import gpitch
import h5py
import pickle


#data_path = "c4dm-01/MAPS_original/AkPnBcht/ISOL/NO/"
#save_path = "c4dm-04/alvarado/results/"

train_data_path = "/media/pa/TOSHIBA EXT/Datasets/MAPS/AkPnBcht/ISOL/NO/"
test_data_path = "/media/pa/TOSHIBA EXT/Datasets/MAPS/AkPnBcht/MUS/"


def load_traindata(path, pitch, frames=88200, scaled=True):

    path = path + train_data_path
    lfiles = gpitch.methods.load_filenames(directory=path, pattern='F', pitches=pitch)
    nfiles = len(lfiles)
    data = nfiles*[None]

    for i in range(nfiles):
        if lfiles[i].find("S1") is not -1:
            start = 30000
        else:
            start = 20000
        data[i] = gpitch.readaudio(path + lfiles[i], start=start, frames=frames, scaled=scaled)
    if nfiles == 1:
        return data[0], lfiles[0]
    else:
        return data, lfiles


def load_testdata(filename, start, frames, window_size):

    # load data
    audio = gpitch.readaudio(fname=test_data_path + filename + '.wav', start=start, frames=frames, scaled=True)

    # segment data
    x, y = gpitch.segmented(x=audio[0], y=audio[1], window_size=window_size)
    return x, y


def init_inducing(x, y, num_sources=1):
    nwin = len(x)
    u = nwin * [None]
    z = nwin*[None]

    for i in range(nwin):
        a, b = gpitch.init_liv(x=x[i], y=y[i], num_sources=num_sources)
        z[i] = a[0][0]
        u[i] = b
    return u, z

def train():
    pass


def write(fname, data):
    pickle.dump(data, open(fname + ".p", "wb"))


def read(fname):
    return pickle.load(fname, open(fname, "rb"))


def writecov(path, fname, data):
    with h5py.File(fname + '.h5', 'w') as hf:
        hf.create_dataset(path + fname, data=data)


def readcov(fname):
    with h5py.File(fname + '.h5', 'r') as hf:
        data = hf[fname][:]
    return data
