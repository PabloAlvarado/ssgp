import numpy as np
from scipy import signal
from gpitch import windowed


def frame(y, window_size, overlap, fs):
    x_b, y_b = windowed.balance_data_size(y, window_size, overlap, fs)
    new_n = x_b.shape
    xout = []
    yout = []
    n = x_b.size
    l = (window_size - overlap)
    nw = (n - overlap) / l
    for i in range(nw):

        if i == 0:
            win = signal.hann(window_size).reshape(-1, 1)
            win[0:l] = 1.

        elif i == nw - 1:
            win = signal.hann(window_size).reshape(-1, 1)
            win[-l:] = 1.

        else:
            win = signal.hann(window_size).reshape(-1, 1)

        xout.append(x_b[i * l: i * l + window_size].copy().reshape(-1, 1))
        yout.append(y_b[i * l: i * l + window_size].copy().reshape(-1, 1) * win.copy())
    return new_n, xout, yout


def merge_x(xl, ws, ov, n):
    x_joint_l = []
    for i in range(len(xl)-1):
        x_joint_l.append(xl[i][0:ws-ov].copy())
    x_joint_l.append(xl[-1][0:].copy())

    x_joint = np.vstack(x_joint_l).reshape(-1, 1)
    x_joint = x_joint[0:n]
    return x_joint


def merge_y(yl, new_n, ws, ov, n):
    yout = np.zeros((new_n))
    nw = len(yl)
    yout[0:ws - ov] = yl[0][0:ws - ov].copy()
    yout[-(ws - ov):] = yl[-1][-(ws - ov):].copy()
    for i in range(1, nw):
        yout[i * (ws - ov): i * (ws - ov) + ov] = (yl[i][0:ov] + yl[i - 1][-ov:])
    yout = yout[0:n].copy()
    return yout
