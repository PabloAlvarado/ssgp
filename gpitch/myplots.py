import matplotlib.pyplot as plt
import numpy as np
from gpitch import logistic
from scipy.fftpack import fft
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_zoom_in(model, figsize=(16, 4), source=1, width="30%", height="30%", loc=3,
                 limits=(8.10, 8.107, -0.47, 0.32), loc_mark=(2, 4),
                 bbox_to_anchor=(0.5, 0, 1, 1)):

    fig, ax = plt.subplots(figsize=figsize)  # create a new figure with a default 111 subplot
    ax.plotgp = plotgp
    ax.plotgp(x=model.real_src[source].x,
              y=model.real_src[source].y,
              xnew=model.test_data.x,
              mean=model.esource[source][0],
              variance=model.esource[source][1])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.xlim(0.0, 14.0)
    plt.tight_layout()
    plt.legend(['True source', 'GP estimate', 'Uncertainty'])

    # create inside segment
    # zoom_in = inset_axes(ax, width=width, height=height, loc=loc)
    zoom_in = inset_axes(ax, width=width, height=height, loc=loc, bbox_to_anchor=bbox_to_anchor,
                         bbox_transform=ax.transAxes)
    zoom_in.plotgp = plotgp

    # plot in segment
    zoom_in.plotgp(x=model.real_src[source].x,
                   y=model.real_src[source].y,
                   xnew=model.test_data.x,
                   mean=model.esource[source][0],
                   variance=model.esource[source][1])

    x1, x2, y1, y2 = limits  # specify the limits
    zoom_in.set_xlim(x1, x2)  # apply the x-limits
    zoom_in.set_ylim(y1, y2)  # apply the y-limits
    plt.yticks([])  # turn off ticks in segment
    plt.xticks([])  #
    mark_inset(ax, zoom_in, loc1=loc_mark[0], loc2=loc_mark[1], fc="none", ec="0.5")  # put zoom in marks


def plotgp(x, y, xnew, mean, variance):
    """
    Plot gaussian process

    """
    plt.plot(x, y, '.k', ms=3, mew=1)
    plt.plot(xnew, mean, 'C0', lw=2)
    plt.fill_between(xnew[:, 0],
                     mean[:, 0] - 2*np.sqrt(variance[:, 0]),
                     mean[:, 0] + 2*np.sqrt(variance[:, 0]),
                     color='C0', alpha=0.25)


def plot_predict(x, mean, var, z, nlinfun=logistic, latent=False, plot_z=True, plot_latent=True):
    """Basic plot unit for ploting predictions in general"""
    if latent:
        plt.plot(x, nlinfun(mean), 'C0', lw=2)
        plt.fill_between(x[:, 0], nlinfun(mean[:, 0] - 2*np.sqrt(var[:, 0])),
                         nlinfun(mean[:, 0] + 2*np.sqrt(var[:, 0])), color='C0', alpha=0.2)

        if plot_latent:
            plt.twinx()

            plt.plot(x, mean, 'C2', lw=2, alpha=0.5)
            plt.fill_between(x[:, 0], mean[:, 0] - 2*np.sqrt(var[:, 0]),
                             mean[:, 0] + 2*np.sqrt(var[:, 0]), color='C2', alpha=0.1)
    else:
        plt.plot(x, mean, 'C0', lw=2)
        plt.fill_between(x[:, 0], mean[:, 0] - 2*np.sqrt(var[:, 0]),
                         mean[:, 0] + 2*np.sqrt(var[:, 0]), color='C0', alpha=0.2)

    if plot_z:
        plt.plot(z, -0.01 + 0.*z, '|k', mew=1)


# PLOTS EVALUATION__________________________________________________________________________________________________
def plot_data(x, y, source=None, maxncol=4):
    if source is None:
        num_sources = 0
    else:
        num_sources = len(source)

    if num_sources < maxncol:
        ncol = num_sources

    else:
        ncol = maxncol

    if num_sources == 0:
        ncol = 1
        nrow = 1
    else:
        nrow = 2 + int((num_sources-1)/maxncol)

    plt.figure(figsize=(16, 4*nrow))
    plt.subplot(nrow, ncol, (1, ncol))
    plt.plot(x, y)
    plt.xlim(x[0], x[-1])
    plt.legend(["Data"], loc=1)

    if source is not None:
        for i in range(num_sources):
            plt.subplot(nrow, ncol, i + 1 + ncol)
            plt.plot(x, source[i])
            plt.xlim(x[0], x[-1])
            plt.legend(["Source " + str(i + 1)], loc=1)


def plot_predict_all(x, mean_act, var_act, mean_com, var_com, m, nlinfun=logistic):
    num_sources = len(mean_act)
    ncol = 4
    nrow = 2*(1 + int((num_sources-1)/ncol))
    plt.figure(figsize=(16, 4*2*(num_sources/ncol)))
    for i in range(num_sources):
        plt.subplot(nrow, ncol, ncol*(i/ncol) + i + 1)
        plot_predict(x, mean_act[i], var_act[i], m.za[i].value, nlinfun, True)

        plt.subplot(nrow, ncol, ncol*(i/ncol + 1) + i + 1)
        plot_predict(x, mean_com[i], var_com[i], m.zc[i].value, nlinfun)


def plot_sources_all(x, y, esource, source=None, maxncol=4, fignumber=100):
    if esource is None:
        num_sources = 0
    else:
        num_sources = len(esource)

    if num_sources < maxncol:
        ncol = num_sources

    else:
        ncol = maxncol

    if num_sources == 0:
        ncol = 1
        nrow = 1
    else:
        nrow = 2 + int((num_sources-1)/maxncol)

    all_prediction = np.zeros((x.size, 1))
    for i in range(num_sources):
        all_prediction += esource[i]

    plt.figure(fignumber, figsize=(16, 4*nrow))
    plt.subplot(nrow, ncol, (1, ncol))
    plt.plot(x, y, 'xk')
    plt.plot(x, all_prediction, lw=2)
    plt.ylim(-1.1, 1.1)
    # plt.xlim(x[0], x[-1])
    plt.legend(["Data"], loc=1)

    for i in range(num_sources):
        plt.subplot(nrow, ncol, i + 1 + ncol)
        if source is not None:
            plt.plot(x, source[i], 'xk')
        plt.plot(x, esource[i], lw=2)
        plt.ylim(-1.1, 1.1)
        # plt.legend(["Real source " + str(i + 1), "Estimated source " + str(i + 1)], loc=1)
        # plt.xlim(x[0], x[-1])


# TRAINING PLOTS

def plot_training_all(x, y, source, m_a, v_a, m_c, v_c, m):
    plt.figure(figsize=(16, 3))

    plt.subplot(1, 4, 1), plt.title('data')
    plt.plot(x, y), plt.ylim([-1.1, 1.1])

    plt.subplot(1, 4, 2), plt.title('approximation')
    plt.plot(x, source), plt.ylim([-1.1, 1.1])

    plt.subplot(1, 4, 3), plt.title('activation')
    plot_predict(x, m_a, v_a, m.za[0].value, nlinfun=logistic, latent=True)

    plt.subplot(1, 4, 4), plt.title('component')
    plot_predict(x, m_c, v_c, m.zc[0].value, nlinfun=logistic)


def plot_trained_models(m, instr_name):
    for i in range(len(m)):
        x = m[i].x.value.copy()
        y = m[i].y.value.copy()

        mean_g, var_g = m[i].prediction_act
        mean_f, var_f = m[i].prediction_com

        source = logistic(mean_g)*mean_f

        plot_training_all(x, y, source, mean_g, var_g, mean_f, var_f, m[i])

    plt.suptitle(instr_name)


def plot_fft(F1, F2, y, y_k, numf, iparam):
    plt.figure(figsize=(16, 16))  # visualize loaded data
    for i in range(numf):
        Y1 = np.abs(fft(y[i].reshape(-1,)))[0:16000]
        Y2 = np.abs(fft(y_k[i].reshape(-1,)))[0:5*16000]
        Y1 /= np.max(Y1)
        Y2 /= np.max(Y2)

        plt.subplot(4, 3, i+1)
        plt.plot(F1, Y1, 'C0')
        plt.plot(F2, Y2, 'C1')
        plt.twinx()
        plt.plot(iparam[i][0], iparam[i][1] / np.max(iparam[i][1]), '|C4', mew=2)
        plt.xlim(0, 4000)


def plot_parameters(m):
    plt.figure(figsize=(16, 4))
    nr, nc = 1, 5
    plt.subplot(nr, nc, 1), plt.title('lengthscale activation'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_act[0].lengthscales.value, '.C1')
    plt.xlim(-1, 12)  # plt.ylim([0, 2.5])

    plt.subplot(nr, nc, 2), plt.title('variance activation'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_act[0].variance.value, '.C1')
    plt.xlim(-1, 12)  # plt.ylim([0, 5.])

    plt.subplot(nr, nc, 3), plt.title('lengthscale component'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_com[0].lengthscales.value, 'C1.')
    plt.xlim(-1, 12)  # plt.ylim([0, 5.])

    plt.subplot(nr, nc, 4), plt.title('f0 component'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].kern_com[0].frequency[0].value, 'C1.')
    plt.xlim(-1, 12)  # plt.ylim([200, 500])

    plt.subplot(nr, nc, 5), plt.title('noise variance'), plt.grid(True)
    for i in range(len(m)):
        plt.plot(i, m[i].likelihood.variance.value, 'C1.')
    plt.xlim(-1, 12)  # plt.ylim([-0.001, 0.005])


# EXTRA PLOTS
def plot_pdgp(x, m, list_predictions, nlinfun=logistic):
    m_a, v_a, m_c, v_c, source = list_predictions

    ncol, nrow = 1, 6
    for i in range(3):
        plt.subplot(nrow, ncol, i+1), plt.title('activation ' + str(i+1))
        plot_predict(x, m_a[i], v_a[i], m.za[i].value, nlinfun=nlinfun, latent=True, plot_latent=False)

    for i in range(3):
        plt.subplot(nrow, ncol, i+4), plt.title('component ' + str(i+1))
        plot_predict(x, m_c[i], v_c[i], m.zc[i].value, nlinfun=nlinfun, latent=False)

#     for i in range(3):
#         plt.subplot(nrow, ncol, i+7), plt.title('source '+ str(i+1))
#         plt.plot(x, source[i])

#     plt.subplot(nrow, ncol, 10), plt.title('data and prediction')
#     plt.plot(x, y, 'C0')
#     plt.plot(x, source[0] + source[1] + source[2], 'C1')
