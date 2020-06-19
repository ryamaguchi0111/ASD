import numpy as np
from scipy import signal


def get_ricker_wave(t, s=1):
    '''
    https://en.wikipedia.org/wiki/Mexican_hat_wavelet
    '''
    return 2 * (1-(t/s)**2) * np.e**(-t**2/(2*s**2)) / (np.sqrt(3*s)*np.pi**0.25)

def ricker(width, a):
    x = (np.arange(width) - int(width / 2)) / a
    return get_ricker_wave(x)

def sumple_chirp():
    return signal.chirp(np.arange(0, 10, 1/44100), f0=10, f1=4000, t1=10)

def step_cwt(x, mw, A, step=1):
    """
    continuous wavelet transform with regulation of convolution step.

    Parameters
    ----------
    x: list[froat]
        input signal
    mw: function
        mother wavelet
        plz input 'ricker'
    A: list[int]
        scale
    step: int 
        frequency of convolution
    
    Returns
    -------
    wavelet_metrix: ndarray
        [scale: int, wavelet_transformed_value: float]

    Examples
    --------
    >>> import cwt
    >>> import matplotlib.pyplot as plt
    >>> cwtmatr = cwt.step_cwt(cwt.sumple_chirp(), cwt.ricker, [1, 2, 3, 4, 5, 6], step=10)
    >>> plt.imshow(cwtmatr, extent=[0, 10, 1, 6], aspect='auto')

    output.shape is proportional to 'step' and 'A'
    >>> cwt.sumple_chirp.shape
    (441000,)
    >>> cwt.step_cwt(cwt.sumple_chirp(), cwt.ricker, [1, 2, 3, 4, 5, 6], step=1).shape
    (6, 441000)
    >>> cwt.step_cwt(cwt.sumple_chirp(), cwt.ricker, [1, 2, 3, 4, 5, 6], step=100).shape
    (6, 4410)
    """
    rows = []
    for a in A:
        wave = mw(min(10*a, len(x)), a)
        if step==1:
            rows.append(np.convolve(x, wave, mode='same'))
        else:
            n_wave = len(wave)
            n_step = len(x) // step - 1
            columns = []
            padding_l = np.zeros([n_wave//2])
            padding_r = np.zeros([(n_wave-1)//2])
            _x = np.hstack([padding_l, x, padding_r])
            for i in range(n_step):
                cell = np.sum(_x[step*i:step*i+n_wave] * wave[::-1])
                columns.append(cell)
            rows.append(columns)
    return np.array(rows)

def cwt(x, mw, A):
    return step_cwt(x, mw, A, step=1)