import numpy as np
from numpy.lib.stride_tricks import as_strided
from stft import stft, istft
#import dsdtools
import musdb
from settings import *
from resampy import resample

eps = np.spacing(1)

#dsd = dsdtools.DB(root_dir="/Users/flucoma/datasets/DSD100/")
mus = musdb.DB(root_dir=dataset_path)

tracks = mus.load_mus_tracks(subsets="train")

n = 0
X = None
Y = None
for t in tracks:
    print(t.path)
    mix = resample(np.mean(t.audio,1), 44100, 22050)

    vocals = resample(np.mean(t.targets["vocals"].audio, 1), 44100, 22050)
    drums = resample(np.mean(t.targets["drums"].audio, 1), 44100, 22050)
    bass = resample(np.mean(t.targets["bass"].audio, 1), 44100, 22050)
    other = resample(np.mean(t.targets["other"].audio, 1), 44100, 22050)
    M = np.abs(stft(mix.reshape(-1,1), hop_size, win_size, fft_size)).T
    V = np.abs(stft(vocals.reshape(-1,1), hop_size, win_size, fft_size)).T
    D = np.abs(stft(drums.reshape(-1,1), hop_size, win_size, fft_size)).T
    B = np.abs(stft(bass.reshape(-1,1), hop_size, win_size, fft_size)).T
    O = np.abs(stft(other.reshape(-1,1), hop_size, win_size, fft_size)).T

    ALL = V+D+B+O+eps
    VM = V / ALL
    DM = D / ALL
    BM = B / ALL
    OM = O / ALL

    spec_frames, n_bins = M.shape
    pad_size = int((n_frames -1)/2)
    M = np.concatenate((np.zeros((pad_size,n_bins)), M, np.zeros((pad_size,n_bins))))
    new_strides = (M.strides[0], M.strides[0], M.strides[1])
    M = as_strided(M, (spec_frames, n_frames, n_bins), new_strides).copy()
    for f in range(spec_frames):
        ftr = M[f,np.newaxis,:,:]
        tgt = np.hstack((VM[f,:],DM[f,:],BM[f,:], OM[f,:]))
        np.save(analysis_path+"/"+str(n)+"_X.npy",ftr)
        np.save(analysis_path+"/"+str(n)+"_Y.npy",tgt)
        n = n + 1
