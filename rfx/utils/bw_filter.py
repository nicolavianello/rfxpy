import numpy as np
import scipy as sp
from scipy import signal


def bw_filter(data, freq, fs, ty, order=5):
    ny = 0.5 * fs
    if np.size(freq) == 1:
        fr = freq / ny
        b, a = sp.signal.butter(order, fr, btype=ty)
    else:
        frL = freq[0] / ny
        frH = freq[1] / ny
        b, a = sp.signal.butter(order, [frL, frH], btype=ty)
    y = sp.signal.filtfilt(b, a, data)
    return y
