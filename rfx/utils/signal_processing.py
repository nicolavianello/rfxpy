import numpy as np
from scipy.interpolate import UnivariateSpline


def correlation(x, y):
    """
    Compute the cross correlation function, return the lag x-values, the
    normalized correlation and the correlation time defined as where it assumes 1/e
    values of the maximum
    Args:
        x: first array
        y: second array. Can be the same and then computes the auto-correlation time

    Returns:
        tuple with lag, correlation and correlation time
    """

    lag = np.arange(x.size, dtype="float") - x.size / 2.0
    c = np.correlate(x, y, mode="same")
    c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    S = UnivariateSpline(lag, c - 1 / np.exp(1), s=0)
    tac = S.roots()[S.roots() > 0][0]
    return lag, c, tac
