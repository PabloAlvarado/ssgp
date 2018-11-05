import numpy as np
import pickle
import h5py
import gpitch
import scipy.io
import matplotlib.pyplot as plt
from gpitch.audio import Audio
from scipy import fftpack
from myplots import plotgp
from sklearn.metrics import mean_squared_error as mse
from gpitch import window_overlap


class SoSp:
    """
    Source separation model class
    """

    def __init__(self, instrument, frames, pitches=None, gpu='0', load=True):

        # init session
        self.sess, self.path = gpitch.init_settings(visible_device=gpu)

        self.instrument = instrument
        self.pitches = pitches

        self.train_path = "/import/c4dm-04/alvarado/datasets/ss_amt/training_data/"
        self.test_path = "/import/c4dm-04/alvarado/datasets/ss_amt/test_data/"
        self.kernel_path = '/import/c4dm-04/alvarado/results/sampling_covariance/icassp19/'

        self.train_data = [None]
        self.test_data = Audio()
        self.real_src = []
        self.params = [[], [], []]
        self.kern_sampled = [None]
        self.inducing = [None]
        self.kern_pitches = [None]
        self.model = None
        self.sampled_cov = [None]

        self.mean = []
        self.var = []
        self.smean = []
        self.svar = []

        self.esource = None

        self.load_train()
        self.load_test(frames=frames)

        nrow = len(self.pitches)
        ncol = len(self.test_data.Y)
        self.matrix_var = np.zeros((nrow, ncol))

        self.init_kernel(load=load)
        self.init_model()

    def load_train(self, train_data_path=None):

        if train_data_path is not None:
            self.train_path = train_data_path

        lfiles = gpitch.methods.load_filenames(directory=self.train_path, pattern=self.instrument, pitches=self.pitches)
        nfiles = len(lfiles)
        data = []

        for i in range(nfiles):
            data.append(Audio(path=self.train_path, filename=lfiles[i], frames=32000))
        self.train_data = data

    def load_test(self, window_size=2001, start=0, frames=-1, test_data_path=None):

        # test_file = gpitch.methods.load_filenames(directory=self.test_path, pattern=self.instrument + "_mixture")[0]

        if test_data_path is not None:
            self.test_path = test_data_path

        # self.test_data = Audio(path=self.test_path, filename=test_file, start=start, frames=frames,
        #                        window_size=window_size, scaled=True)

        self.test_data = Audio(window_size=window_size)

        names = ['_C_', '_E_', '_G_']
        for i in range(3):
            source_file = gpitch.methods.load_filenames(directory=self.test_path, pattern=self.instrument + names[i])[0]
            self.real_src.append(Audio(path=self.test_path, filename=source_file, start=start, frames=frames,
                                 window_size=window_size, scaled=False))

        self.test_data.x = self.real_src[0].x.copy()
        self.test_data.y = self.real_src[0].y.copy() + self.real_src[1].y.copy() + self.real_src[2].y.copy()

        if self.test_data.y.size == 16000*14:
            self.test_data.y = np.vstack((self.test_data.y, np.zeros((1, 1)) ))
            self.test_data.x = np.linspace(0.,
                                           (self.test_data.y.size - 1.)/self.test_data.fs,
                                           self.test_data.y.size).reshape(-1, 1)
            # print self.test_data.fs

        self.test_data.windowed()

    def plot_traindata(self, figsize=None):
        nfiles = len(self.train_data)

        if nfiles <= 3:
            ncols = nfiles
        else:
            ncols = 3

        nrows = int(np.ceil(nfiles/3.))

        if figsize is None:
            figsize = (16, 3*nrows)

        plt.figure(figsize=figsize)
        for i in range(nfiles):
            plt.subplot(nrows, ncols, i+1)
            plt.plot(self.train_data[i].x, self.train_data[i].y)
            plt.legend([self.train_data[i].name[9:-10]])
        plt.suptitle("train data " + self.instrument)

    def plot_testdata(self, figsize=(16, 2*3)):
        plt.figure(figsize=figsize)

        plt.subplot(2, 3, (1, 3))
        plt.suptitle("test data " + self.instrument)
        plt.plot(self.test_data.x, self.test_data.y)
        plt.legend([self.test_data.name])

        for i in range(3):
            plt.subplot(2, 3, i + 4)
            plt.plot(self.real_src[i].x, self.real_src[i].y)
            plt.legend([self.real_src[i].name[9:-4]])

    def plot_kernel(self, figsize=None):
        nfiles = len(self.train_data)

        if nfiles <= 3:
            ncols = nfiles
        else:
            ncols = 3

        nrows = int(np.ceil(nfiles / 3.))

        x0 = np.array(0.).reshape(-1, 1)
        x1 = np.linspace(0., 0.01, 441).reshape(-1, 1)

        if figsize is None:
            figsize = (16, 3*nrows)

        plt.figure(figsize=figsize)
        plt.suptitle("sampled kernels")
        for i in range(nfiles):
            plt.subplot(nrows, ncols, i + 1)

            # plt.plot(self.kern_sampled[0][i], self.kern_sampled[1][i])
            # plt.plot(self.kern_sampled[0][i], self.kern_pitches[i].compute_K(self.kern_sampled[0][i], x0))
            plt.plot(self.kern_pitches[i].compute_K(x1, x0))

            plt.title(self.train_data[i].name[18:-13])
            plt.legend(['full kernel', 'approx kernel'])

    def load_kernel(self):
        path = self.kernel_path
        param_filename = gpitch.load_filenames(directory=path, pattern=self.instrument, pitches=self.pitches,
                                               ext='hyperparams.p')

        self.params = [[], [], []]
        self.kern_sampled = [[], []]

        for i in range(len(self.pitches)):
            aux_param = pickle.load(open(path + param_filename[i], "rb"))
            self.params[0].append(aux_param[1])  # lengthscale
            self.params[1].append(aux_param[2])  # var
            self.params[2].append(aux_param[3])  # freq

            self.kern_sampled[0].append(aux_param[4])  # time vector
            self.kern_sampled[1].append(aux_param[5])  # sampled kernel

    def init_kernel(self, covsize=441, num_sam=10000, max_par=1, train=False, save=False, load=False):

        nfiles = len(self.train_data)
        self.params = [[], [], []]
        skern, xkern = nfiles * [np.zeros((1, 1))], nfiles * [None]

        if train:
            scov, samples = nfiles * [None], nfiles * [None]
            self.sampled_cov = nfiles * [None]

            for i in range(nfiles):

                # sample cov matrix
                self.sampled_cov[i], skern[i], samples[i] = gpitch.samplecov.get_cov(self.train_data[i].y,
                                                                                     num_sam=num_sam, size=covsize)

                # approx kernel
                params = gpitch.kernelfit.fit(kern=skern[i], audio=self.train_data[i].y,
                                              file_name=self.train_data[i].name, max_par=max_par)[0]
                self.params[0].append(params[0])  # lengthscale
                self.params[1].append(params[1])  # variances
                self.params[2].append(params[2])   # frequencies

                xkern[i] = np.linspace(0., (covsize - 1.) / self.train_data[i].fs, covsize).reshape(-1, 1)
            self.kern_sampled = [xkern, skern]

            if save:
                self.save()

        elif load:
            self.load_kernel()  # load already learned parameters

        else:
            # init kernels with fft of data
            for i in range(nfiles):
                f0 = gpitch.find_ideal_f0([self.train_data[i].name])[0]

                params = gpitch.init_cparam(y=self.train_data[i].y.copy(),
                                            fs=self.train_data[i].fs,
                                            maxh=max_par,
                                            ideal_f0=f0)

                self.params[0].append(np.array(0.1))  # lengthscale
                self.params[1].append(params[1])  # variances
                self.params[2].append(params[0])  # frequencies

                skern[i] = fftpack.ifft(np.abs(fftpack.fft(self.train_data[i].y.copy().reshape(-1, ))))[0:covsize].real
                skern[i] /= np.max(skern[i])
                xkern[i] = np.linspace(0., (covsize - 1.) / self.train_data[i].fs, covsize).reshape(-1, 1)
            self.kern_sampled = [xkern, skern]

        # init kernel specific pitch
        self.kern_pitches = gpitch.init_kernels.init_kern_com(num_pitches=len(self.train_data),
                                                              lengthscale=self.params[0],
                                                              energy=self.params[1],
                                                              frequency=self.params[2],
                                                              len_fixed=True)

    def init_inducing(self):
        nwin = len(self.test_data.X)
        u = nwin * [None]
        z = nwin * [None]

        for i in range(nwin):
            a, b = gpitch.init_liv(x=self.test_data.X[i], y=self.test_data.Y[i], num_sources=1)
            z[i] = a[0][0][::1]  # use extrema as inducing variables
            u[i] = b[::1]

            # z[i] = self.test_data.X[i].copy() # use all data as inducing variables
            # u[i] = self.test_data.Y[i].copy()
        self.inducing = [z, u]

    def init_model(self):
        """Hi"""
        self.init_inducing()  # init inducing points

        # init model kernel
        kern_model = np.sum(self.kern_pitches)

        # init gp model
        x_init = self.test_data.X[0].copy()
        y_init = self.test_data.Y[0].copy()
        z_init = self.inducing[0][0].copy()
        self.model = gpitch.sgpr_ss.SGPRSS(X=x_init, Y=y_init, kern=kern_model, Z=z_init)

    def reset_model(self, x, y, z):
        self.model.X = x.copy()
        self.model.Y = y.copy()
        self.model.Z = z.copy()
        self.model.likelihood.variance = 1.
        # self.model.likelihood.variance = 0.0001
        # self.model.likelihood.variance.fixed = True

        for i in range(len(self.pitches)):
            # self.model.kern.kern_list[i].kern_list[0].variance = 1.
            # self.model.kern.kern_list[i].kern_list[0].lengthscales = self.params[0][i].copy()
            self.model.kern.kern_list[i].variance = 1.
            self.model.kern.kern_list[i].lengthscales = self.params[0][i].copy()

    def optimize(self, maxiter=1000, disp=1, nwin=None):

        self.mean = []
        self.var = []
        self.smean = []
        self.svar = []

        if nwin is None:
            nwin = len(self.test_data.Y)

        for i in range(nwin):

            # reset model
            self.reset_model(x=self.test_data.X[i],
                             y=self.test_data.Y[i],
                             z=self.inducing[0][i])

            # optimize window
            print("optimizing window " + str(i))
            self.model.optimize(disp=disp, maxiter=maxiter)

            # save learned params
            for j in range(len(self.pitches)):
                self.matrix_var[j, i] = self.model.kern.kern_list[j].variance.value.copy()
                # self.matrix_var[j, i] = self.model.kern.kern_list[j].kern_list[0].variance.value.copy()

            # predict mixture function
            mean, var = self.model.predict_f(self.test_data.X[i].copy())
            self.mean.append(mean)
            self.var.append(var)

            # predict sources
            smean, svar = self.model.predict_s(self.test_data.X[i].copy())
            self.smean.append(smean)
            self.svar.append(svar)

    def save(self):
        # save results
        for i in range(len(self.pitches)):
            auxname = self.train_data[i].name.strip('.wav')
            fname_cov = auxname + '_cov_matrix'
            fname_param = self.path + self.kernel_path + auxname + '_kern_params'

            with h5py.File(self.path + self.kernel_path + fname_cov + '.h5', 'w') as hf:
                hf.create_dataset(fname_cov, data=self.sampled_cov[i])

            pickle.dump([self.params[0][i],
                         self.params[1][i],
                         self.params[2][i],
                         self.kern_sampled[0][i],
                         self.kern_sampled[1][i]],
                        open(fname_param + ".p", "wb"))

    def predict_f(self, xnew=None):
        if xnew is None:
            mean = np.asarray(self.mean).reshape(-1, 1)
            var = np.asarray(self.var).reshape(-1, 1)
        else:
            mean, var = self.model.predict_f(xnew)

        return mean, var

    def predict_s(self):
        m1, m2, m3 = [], [], []
        for i in range(len(self.smean)):
            m1.append(self.smean[i][0])
            m2.append(self.smean[i][1])
            m3.append(self.smean[i][2])
        # m1 = np.asarray(m1).reshape(-1, 1)
        # m2 = np.asarray(m2).reshape(-1, 1)
        # m3 = np.asarray(m3).reshape(-1, 1)
        ws_aux = 2001
        n_aux = self.test_data.x.size
        m1 = window_overlap.merged_mean(y=m1, ws=ws_aux, n=n_aux)
        m2 = window_overlap.merged_mean(y=m2, ws=ws_aux, n=n_aux)
        m3 = window_overlap.merged_mean(y=m3, ws=ws_aux, n=n_aux)

        v1, v2, v3 = [], [], []
        for i in range(len(self.smean)):
            v1.append(self.svar[i][0])
            v2.append(self.svar[i][1])
            v3.append(self.svar[i][2])
        # v1 = np.asarray(v1).reshape(-1, 1)
        # v2 = np.asarray(v2).reshape(-1, 1)
        # v3 = np.asarray(v3).reshape(-1, 1)
        v1 = window_overlap.merged_variance(y=v1, ws=ws_aux, n=n_aux)
        v2 = window_overlap.merged_variance(y=v2, ws=ws_aux, n=n_aux)
        v3 =  window_overlap.merged_variance(y=v3, ws=ws_aux, n=n_aux)


        if m1.size == 224001:
            m1 = m1[0:-1].reshape(-1, 1)
            m2 = m2[0:-1].reshape(-1, 1)
            m3 = m3[0:-1].reshape(-1, 1)

            v1 = v1[0:-1].reshape(-1, 1)
            v2 = v2[0:-1].reshape(-1, 1)
            v3 = v3[0:-1].reshape(-1, 1)
            self.test_data.x = self.test_data.x[0:-1].reshape(-1, 1)
            self.test_data.y = self.test_data.y[0:-1].reshape(-1, 1)

        self.esource = [[m1, v1], [m2, v2], [m3, v3]]  # estimated sources

    def plot_results(self, figsize=(16, 3*4)):

        plt.figure(figsize=figsize)
        plt.subplot(4, 1, 1)
        plt.suptitle("test data " + self.instrument)
        plt.plot(self.test_data.x, self.test_data.y)
        plt.legend([self.test_data.name])

        for i in range(3):
            plt.subplot(4, 1, i + 2)
            plotgp(x=self.real_src[i].x, y=self.real_src[i].y,
                   xnew=self.real_src[i].x,
                   mean=self.esource[i][0], variance=self.esource[i][1])
            # plt.plot(self.real_src[i].x, self.real_src[i].y)
            # plt.plot(self.real_src[i].x, self.esource[i])
            plt.legend([self.real_src[i].name[9:-4]])
            # plt.ylim(-1., 1.)

        # # Three subplots sharing both x/y axes
        # f, ax = plt.subplots(4, sharex=True, sharey=True, figsize=(16, 3*4))
        # ax[0].plot(self.test_data.x, self.test_data.y)
        # ax[0].set_title('Sharing both axes')
        # for i in range(3):
        #
        #     ax[i + 1].plot(self.real_src[i].x, self.real_src[i].y)
        #     ax[i + 1].plot(self.real_src[i].x, self.esource[i])
        #     ax[i + 1].legend([self.real_src[i].name[9:-4]])
        #
        # #f.subplots_adjust(hspace=0)
        # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    def save_results(self):
        source = [self.real_src[0].y, self.real_src[1].y, self.real_src[2].y]
        esource = [self.esource[0][0], self.esource[1][0], self.esource[2][0]]
        vsource = [self.esource[0][1], self.esource[1][1], self.esource[2][1]]
        scipy.io.savemat("metrics/" + self.instrument + ".mat", {'src': source, 'esrc': esource, 'vsrc': vsource})

    def compute_rmse(self):
        list_mse = []
        num_sources = len(self.esource)
        for i in range(num_sources):
            list_mse.append(np.sqrt(mse(y_true=self.real_src[i].y, y_pred=self.esource[i][0])))
        return np.mean(list_mse)

