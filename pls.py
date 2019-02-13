# This program performs preprocess on the data and then predictes the values usin pls.

# Programmer: Mehrdad Kashefi

# Import libraries
import scipy.io as sio
import numpy as np
import scipy.signal as sig
# Loading Data
mat = sio.loadmat('/home/mehrdad/Datasets/rat_force/rat1.mat')
force = mat['Force_new_pre']
lfp = mat['LFP_new_pre']

fs = 1000
num_channel = 16
filter_degree = 4

# Shuffling trials
index = np.random.permutation(lfp.shape[2])  # Num trials ~ 74
lfp = lfp[:, :, index]
force = force[index, :]

# Filtering below 1 Hz
[b, a] = sig.butter(filter_degree, 1/(fs/2), btype='high', analog=False, output='ba')
# Filtering and appling commen average filter CAR
for trial in range(lfp.shape[2]):
    lfp[:, :, trial] = sig.filtfilt(b, a, lfp[:, :, trial].T).T

