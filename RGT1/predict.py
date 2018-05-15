import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from torch.autograd import Variable
from stft import stft, istft
import musdb
from settings import *
from resampy import resample
from model import CNN
from scipy.io import wavfile

eps = np.spacing(1)
mus = musdb.DB(root_dir = dataset_path)
tracks = mus.load_mus_tracks(subsets = "Test")
model = CNN()
model.load_state_dict(torch.load("cnn.pickle"))
if torch.cuda.is_available():
       model.cuda()

model.train(False)

for t in tracks:
    filename = t.path.split("/")[-1]
    print(filename)
    mix = resample(np.mean(t.audio,1), 44100, 22050)
    M = stft(mix.reshape(-1,1), hop_size, win_size, fft_size)
    Mmag = np.abs(M)

    Mmag = Mmag.T
    spec_frames, n_bins = Mmag.shape
    pad_size = int((n_frames -1)/2)
    Mmag = np.concatenate((np.zeros((pad_size,n_bins)), Mmag, np.zeros((pad_size,n_bins))))
    new_strides = (Mmag.strides[0], Mmag.strides[0], Mmag.strides[1])
    Mmag = as_strided(Mmag, (spec_frames, n_frames, n_bins), new_strides)
    Mmag = Mmag[:,np.newaxis,:,:]

    vocals = np.zeros(M.T.shape)
    bass = np.zeros(M.T.shape)
    drums = np.zeros(M.T.shape)
    other = np.zeros(M.T.shape)

    for i in range(spec_frames):
        X = Mmag[i,:,:,:]
        in_data = torch.from_numpy(X.astype(np.float32)[np.newaxis,:,:,:])
        if torch.cuda.is_available():
           in_data = in_data.cuda()
        i_result = model(Variable(in_data)).cpu().data.numpy()
        vocals[i,:] = i_result[0,:n_bins]
        drums[i,:] = i_result[0,n_bins:2*n_bins]
        bass[i,:] = i_result[0,2*n_bins:3*n_bins]
        other[i,:] = i_result[0,3*n_bins:4*n_bins]

    all_masks = vocals + bass + drums + other

    vocals = vocals / all_masks
    bass = bass / all_masks
    drums = drums / all_masks
    other = other / all_masks

    sr = 22050

    np.save("results/vocals/" + filename+"_mask", vocals.T)
    vocal_est = istft(M * vocals.T, hop_size, win_size, sr)
    wavfile.write("results/vocals/" + filename+"_target.wav", sr, vocal_est)

    np.save("results/bass/" + filename+"_mask", bass.T)
    bass_est = istft(M * bass.T, hop_size, win_size, sr)
    wavfile.write("results/bass/" + filename+"_target.wav", sr, bass_est)

    np.save("results/drums/" + filename+"_mask", drums.T)
    drums_est = istft(M * drums.T, hop_size, win_size, sr)
    wavfile.write("results/drums/" + filename+"_target.wav", sr, drums_est)

    np.save("results/other/" + filename+"_mask", other.T)
    other_est = istft(M * other.T, hop_size, win_size, sr)
    wavfile.write("results/other/" + filename+"_target.wav", sr, other_est)
