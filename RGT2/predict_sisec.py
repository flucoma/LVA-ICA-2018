import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from torch.autograd import Variable
from stft import stft, istft
import musdb
from settings import *
from resampy import resample
from model import CNN

eps = np.spacing(1)
model = CNN()
model.load_state_dict(torch.load("cnn.pickle"))

if torch.cuda.is_available():
       model.cuda()

model.train(False)

def predict_file(track):
    def predict_channel(audio):
        length = np.shape(audio)[0]
        m = resample(audio,44100, 22050)
        M = stft(m.reshape(-1,1), hop_size, win_size, fft_size)
        Mmag = np.abs(M).T
        spec_frames, n_bins = Mmag.shape
        pad_size = int((n_frames -1)/2)
        Mmag = np.concatenate((
              np.zeros((pad_size,n_bins)),
              Mmag,
              np.zeros((pad_size,n_bins)))
        )
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
            vocals[i,:] = np.argmax(i_result,1)==0
            drums[i,:] = np.argmax(i_result,1)==1
            bass[i,:] = np.argmax(i_result,1)==2
            other[i,:] = np.argmax(i_result,1)==3

        vocal_est = resample(istft(M * vocals.T, hop_size, win_size, 22050), 22050, 44100, 0)[:length,:]
        bass_est = resample(istft(M * bass.T, hop_size, win_size, 22050), 22050, 44100, 0)[:length,:]
        drums_est = resample(istft(M * drums.T, hop_size, win_size, 22050), 22050, 44100, 0)[:length,:]
        other_est = resample(istft(M * other.T, hop_size, win_size, 22050), 22050, 44100, 0)[:length,:]
        return (vocal_est,bass_est,drums_est,other_est)

    (v1,b1,d1,o1) = predict_channel(track.audio[:,0])
    (v2,b2,d2,o2) = predict_channel(track.audio[:,1])

    return {
      'vocals':np.hstack((v1,v2)),
      'bass':np.hstack((b1,b2)),
      'drums':np.hstack((d1,d2)),
      'other':np.hstack((o1,o2)),
    }
mus = musdb.DB(root_dir = dataset_path)
mus.run(predict_file, estimates_dir="sisec_estimates")
