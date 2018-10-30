from scipy.signal import hann
import numpy as np


def flex_hann(n, m):
    f = np.ones((n, 1))
    real_size = 2*m + 1
    wp = hann(real_size)
    f[ :m] = wp[ :m].reshape(-1,1)
    f[-m:] = wp[-m:].reshape(-1,1)
    return f

def frame(x, y, window_size, overlap, fs):
    x_b, y_b = balance_data_size(y, window_size, overlap, fs)
    xout = []
    yout = []
    n = x_b.size
    l = (window_size - overlap)
    nw = (n - overlap) / l
    for i in range(nw):
        xout.append(x_b[i*l : i*l + window_size].copy().reshape(-1, 1))
        yout.append(y_b[i*l : i*l + window_size].copy().reshape(-1, 1))
    return xout, yout

def merged_y(y, window_size, overlap):
    l = window_size - overlap
    nw = len(y)
    n = nw*l + overlap
    y_w = []
    for i in range(nw):
        if i == 0 :
            win = flex_hann(window_size, overlap).reshape(-1, 1)
            win[:overlap] = 1.
        elif i == nw-1:
            win = flex_hann(window_size, overlap).reshape(-1, 1)                        
            win[-overlap:] = 1.
        else:
            win = flex_hann(window_size, overlap).reshape(-1, 1)
        y_w.append( y[i].copy()*win.copy() ) 
    yout = np.zeros((n, 1))
    yout[:l] = y_w[0][:l]
    yout[-l:] = y_w[-1][-l:]
    for i in range(nw-1):
        aux = np.zeros((l, 1))
        aux[:overlap] = y_w[i][-overlap:].copy()  
        yout[(i+1)*l : (i+2)*l ] =   aux.copy() + y_w[i+1][:l].copy()
    return yout

def merged_x(x, window_size, overlap):
    l = window_size - overlap
    nw = len(x)
    n = nw*l + overlap
    xout = np.zeros((n, 1))
    xout[:l] = x[0][:l]
    xout[-l:] = x[-1][-l:]
    for i in range(nw-1):
        xout[(i+1)*l : (i+2)*l ] =  x[i+1][:l].copy()
    return xout

def merged_n_trim(x, y, window_size, overlap, num_samples):
    x_aux = merged_x(x=list(x), window_size=window_size, overlap=overlap)
    y_aux = merged_y(y=list(y), window_size=window_size, overlap=overlap)
    x_final = x_aux[0:num_samples].reshape(-1,1)
    y_final = y_aux[0:num_samples].reshape(-1,1)
    return x_final, y_final

def balance_data_size(y, window_size, overlap, fs):
    n = y.size
    n_samples = []
    for i in range(1, 1000):
        n_samples.append(i*(window_size - overlap) + overlap)
    n_samples = np.asarray(n_samples).reshape(-1, )
    idx = np.argmin(np.abs(n - n_samples))
    
    if n <= n_samples[idx]:
        n_new = n_samples[idx].copy()
    else:
        n_new = n_samples[idx + 1].copy()
        
    if n == n_new:
        y_new = y.copy()
        n_new = y_new.size
    else:
        added_n = np.abs(n_new-n)
        y_new = np.vstack( ( y.copy(), np.zeros((added_n, 1)) ) ).reshape(-1, 1)
    
    x_new = np.linspace(0, (n_new - 1.)/fs, n_new).reshape(-1, 1)    
    return x_new, y_new

def merge_all(inlist):
    outlist = [ [[], [], []],
                [[], [], []],
                [[], [], []],
                [[], [], []],
                [[], [], []] ]
    nrow = len(outlist)
    ncol = len(inlist)
    
    for j in range(nrow):
        for i in range(ncol):        
            outlist[j][0].append(inlist[i][j][0])
            outlist[j][1].append(inlist[i][j][1])
            outlist[j][2].append(inlist[i][j][2])
    
    return outlist

def get_results_arrays(x, y, sl, window_size, overlap, num_samples):
    s1 = merged_y(sl[0], window_size, overlap)
    s2 = merged_y(sl[1], window_size, overlap)
    s3 = merged_y(sl[2], window_size, overlap)

    x = merged_x(x, window_size, overlap)
    y = merged_y(y, window_size, overlap)

    s1_trim = s1[0:num_samples].reshape(-1, 1)
    s2_trim = s2[0:num_samples].reshape(-1, 1)
    s3_trim = s3[0:num_samples].reshape(-1, 1)

    x_trim = x[0:num_samples].reshape(-1, 1)
    y_trim = y[0:num_samples].reshape(-1, 1)

    s = [s1_trim, s2_trim, s3_trim]
    return x_trim, y_trim, s