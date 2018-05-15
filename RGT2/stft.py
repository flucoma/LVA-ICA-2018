import numpy as np
from scipy import signal

def stft(x, hop_size, win_size, fft_size = 2048):
    n_frames = int(np.floor(x.shape[0] / hop_size)+ 1)
    half_win = int(win_size / 2)
    x = np.concatenate((np.zeros((half_win,1)), x, np.zeros((half_win,1))))
    size = x.strides[1]
    shape = (n_frames, win_size)
    strides = (size * hop_size, size)
    frames = np.lib.stride_tricks.as_strided(x, shape = shape, strides = strides)
    frames = frames * signal.hanning(win_size, sym = False)
    return np.fft.rfft(frames, fft_size).T / (win_size/4)

def istft(spectrogram, hop_size, win_size, sample_rate):
    window = signal.hanning(win_size, sym = False)
    half_window = int(win_size / 2)
    frames = np.fft.irfft(spectrogram.T * (win_size / 4))
    n_frames = frames.shape[0]
    result_length = ((n_frames - 1) * hop_size) + win_size + hop_size
    result = np.zeros((result_length, 1))
    norm = np.zeros((result_length, 1))
    wsquare = (window * window).reshape(-1, 1)
    for i in range(n_frames):
        indices = i * hop_size + np.r_[0:win_size]
        result[indices] +=  frames[i,:win_size].reshape(-1, 1) * window.reshape(-1, 1)
        norm[indices] += wsquare
    min_float = np.finfo(spectrogram.dtype).tiny
    result /= np.where(norm > min_float, norm, 1.0)
    result = result[half_window:result_length - half_window]
    return result
