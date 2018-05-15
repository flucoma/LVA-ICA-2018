# Improving Single-Network Single-Channel Separation of Musical Audio with Convolutional Layers

This repository contains the code to reproduce the musical audio source separation presented in

> Roma, G., Green, O. & Tremblay, P. A. Improving single-network single-channel separation of musical audio with convolutional layers
14th International Conference on Latent Variable Analysis and Signal Separation (2018)

The original evaluation used the DSD100 dataset via [dsdtools](https://github.com/faroit/dsdtools). For the 2018 SiSEC campaign it was updated to use the [musdb18](https://sigsep.github.io/musdb) dataset.

#### Requirements
- numpy
- scipy
- pytorch
- resampy
- musdb
- mir_eval

#### How to use
The two proposed variabts are in RGT1 and RGT2, corresponding to the SiSEC submissions. The first uses a soft mask and MSE loss, the second uses a 2D softmax loss to predict a binary mask.

In each directory, configure the dataset path and the analysis path (for temporary analysis files).
Then run each script in order as needed:
- python analyze.py
- python train.py
- python predict.py

Using a GPU is advised for the train.py script.

For evaluation, the script eval_track.py will evaluate one track, so it can be run in parallel using e.g. GNU parallel:

parallel -j4 python eval_track.py {} ::: {0..49}

The scripts predict_sisec.py and eval_musdb.py run the evaluation of the Test part of the dataset according to the SiSEC campaign procedure. Since the original approach is single-channel it is run for each channel separately.
