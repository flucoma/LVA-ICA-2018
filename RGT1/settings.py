n_bins = 1025

fft_size = 2 * (n_bins - 1)

hop_size = fft_size // 8

win_size = fft_size

n_frames = 11

batch_size = 500

n_epochs = 100

learning_rate = 0.001

dataset_path = "/path/to/musdb18/"

analysis_path = "/path/to/scrath/space/"

val_size = 0.2

patience = 5
