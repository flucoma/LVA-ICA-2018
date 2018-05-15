import sys
from multiprocessing import Pool, Array
import numpy as np
from  mir_eval.separation import bss_eval_sources
import mir_eval
from settings import *
from resampy import resample
import musdb
from scipy.io import wavfile


eps = np.spacing(1)

def read_wav(path):
	sr, wave = wavfile.read(path)
	return wave

def eval_file(track):
	filename = track.path.split("/")[-1]
	mix = np.mean(track.audio, 1).reshape(-1,1)
	l = mix.shape[0]
	vocals_ref = np.mean(track.targets["vocals"].audio, 1)
	drums_ref = np.mean(track.targets["drums"].audio, 1)
	bass_ref = np.mean(track.targets["bass"].audio, 1)
	other_ref = np.mean(track.targets["other"].audio, 1)


	vocals_est = resample(read_wav("results/vocals/" +filename+"_target.wav") +eps, 22050, 44100)
	drums_est = resample(read_wav("results/drums/" + filename+"_target.wav")+eps, 22050, 44100)
	bass_est = resample(read_wav("results/bass/" + filename+"_target.wav") +eps, 22050, 44100)
	other_est = resample(read_wav("results/other/" + filename+"_target.wav") +eps, 22050, 44100)

	ref = np.vstack((
		vocals_ref[np.newaxis,:],
		drums_ref[np.newaxis,:],
		bass_ref[np.newaxis,:],
		other_ref[np.newaxis,:]
	))

	est = np.vstack((
		vocals_est[np.newaxis,0:ref.shape[1]],
		drums_est[np.newaxis,0:ref.shape[1]],
		bass_est[np.newaxis,0:ref.shape[1]],
		other_est[np.newaxis,0:ref.shape[1]],
	))
	sdr = [0,0,0,0]
	sir = [0,0,0,0]
	sar = [0,0,0,0]
	print(filename)
	sdr, sir, sar, p = bss_eval_sources(ref, est, False)
	result = np.vstack((sdr,sir,sar))
	print(sdr, sir, sar)
	np.save("results/eval/"+filename+"_eval.npy", result)
mus = musdb.DB(root_dir = dataset_path)
tracks = mus.load_mus_tracks(subsets="Test")
eval_file(tracks[int(sys.argv[1])])
