# This program performs preprocess on the data and then predictes the values usin pls.

# Programmer: Mehrdad Kashefi

# Import libraries
import scipy.io as sio
import numpy as np
import scipy.signal as sig
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, explained_variance_score,r2_score
from math import sqrt
import statistics as st
import matplotlib.pyplot as plt

#Costom loss coefficient of correlation (r) and
def LossR(yTrue,yPred):
    Res = sum((yTrue - np.mean(yTrue))* (yPred - np.mean(yPred))) / sqrt(sum((yTrue - np.mean(yTrue))**2)* sum((yPred - np.mean(yPred))**2))
    return Res

# Loading Data
try:
    mat = sio.loadmat('/home/mehrdad/Datasets/rat_force/rat1.mat')
except FileNotFoundError:
    mat = sio.loadmat('/Users/mehrdadkashefi/Datasets/rat_force/rat1.mat')
force = mat['Force_new_pre']
lfp = mat['LFP_new_pre']

fs = 1000
num_channel = 16
filter_degree = 4
downsample_rate = int(np.round(fs/10))
lag = 10
num_fold = 10

# Shuffling trials
# index = np.random.permutation(lfp.shape[2])  # Num trials ~ 74
# lfp = lfp[:, :, index]
# force = force[index, :]

# Filtering below 1 Hz
[b, a] = sig.butter(filter_degree, 1/(fs/2), btype='high', analog=False, output='ba')
# Filtering and appling commen average filter CAR
for trial in range(lfp.shape[2]):  # Over trials
    lfp[:, :, trial] = sig.filtfilt(b, a, lfp[:, :, trial], axis=1, padtype='odd', method='pad', padlen=3*(max(len(a),len(b))-1))
    channel_mean = np.mean(lfp[:, :, trial], 0)
    for channel in range(lfp.shape[0]):  # loop over channels  ---> can improve
        lfp[channel, :, trial] = lfp[channel, :, trial] - channel_mean

lfp = lfp.reshape(lfp.shape[0], lfp.shape[1]*lfp.shape[2], order='F')  # This part should be checked
# force = force.T
force = force.reshape(force.shape[0]*force.shape[1])

num_features = (lag+1)*lfp.shape[0]
# Considered frequency bands
freq_bands = np.array([[1, 4], [4, 8], [8, 12], [12, 30], [30, 60], [60, 120], [120, 200], [200, 400]])
feature_allband = np.zeros((2210, num_features*freq_bands.shape[0]))  # Num point with lag 10 ---> 2210
for band in range(freq_bands.shape[0]):
    if band == 0:
        [b, a] = sig.butter(filter_degree, freq_bands[band, 1]/(fs/2), btype='low', analog=False, output='ba')
    else:
        [b, a] = sig.butter(filter_degree, [freq_bands[band, 0]/(fs / 2), freq_bands[band, 1]/(fs / 2)], btype='band', analog=False, output='ba')

    feature = sig.filtfilt(b, a, lfp, axis=1,padtype='odd',method='pad',padlen=3*(max(len(a), len(b)) -1))
    feature = np.abs(feature)

    for channel in range(feature.shape[0]):
        feature[channel, :] = sig.savgol_filter(feature[channel, :], window_length=299, polyorder=3)  # Window = 300.odd
        feature_mean = np.mean(feature[channel, :])
        feature_std = np.std(feature[channel, :])
        feature[channel, :] = (feature[channel, :] - feature_mean)/feature_std

    # Performing down sampling
    feature = feature[:, 0:feature.shape[1]:downsample_rate]
    feature = feature.T

    feature_tot = np.zeros((feature.shape[0]-lag, lag+1, feature.shape[1]))  # lag+1 is added for matlab compatibility
    for time in range(feature.shape[0]-lag):
        feature_tot[time, :, :] = feature[time:time+lag+1, :]

    feature_tot = feature_tot.reshape((feature_tot.shape[0], feature_tot.shape[1]*feature_tot.shape[2]), order='F')  # Check the reshape later

    feature_allband[:, num_features*band:num_features*band + num_features] = feature_tot
    print("Calculatring for frequency band ", band+1, " for subject ", 1)

# Down sampling force
force = force[0:len(force):downsample_rate]
force = np.delete(force, range(lag))
# ------------------ PLS ---------------------

fold = np.round(np.linspace(0, feature_allband.shape[0], num_fold+1), )
kfold_index = range(feature_allband.shape[0])

for fold_count in range(num_fold):

    index_train = kfold_index
    index_test = range(int(fold[fold_count]), int(fold[fold_count + 1]))
    index_train = np.delete(index_train, index_test)
    # Feature train-test separation
    feature_allband_train = feature_allband[index_train, :]
    feature_allband_test = feature_allband[index_test, :]
    # force train-test separation
    force_train = force[index_train]
    force_test = force[index_test]

    pls = PLSRegression(n_components=10)
    pls.fit(feature_allband_train, force_train)

    prediction = pls.predict(feature_allband_test)

    # Calculate scores
    score = explained_variance_score(force_test, prediction)
    # Corr = LossR(force_test, prediction)

    print("=============================")
    print("predictions for fold ", fold_count+1)
    print("r2 score is ", score)
   # print("Corrolation score is ", Corr)


print('Hello')