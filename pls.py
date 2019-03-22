# This program performs preprocess on the data and then predictes the values usin pls.

# Programmer: Mehrdad Kashefi

# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig
from sklearn.cross_decomposition import PLSRegression
from mlp_mse import *
from softplus_4variable import softplus_4variable
from sklearn.preprocessing import normalize
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.activations import linear
from keras.regularizers import l2

# Costom loss coefficient of correlation (r) and

def LossCorr(yTrue, yPred):
    Res = np.dot((yTrue - np.mean(yTrue)).T, yPred - np.mean(yPred)) / (np.std(yTrue) * np.std(yPred) * len(yTrue))
    return Res

def LossR2(yTrue, yPred):
    yPred = np.reshape(yPred, (len(yPred), 1))
    yTrue = np.reshape(yTrue, (len(yTrue), 1))
    Res = np.var((yTrue-yPred), ddof=1)/np.var(yTrue, ddof=1)
    return Res

# Loading Data


try:
    mat = sio.loadmat('/home/mehrdad/Datasets/rat_force/rat1.mat')
except FileNotFoundError:
    mat = sio.loadmat('/Users/mehrdadkashefi/Datasets/rat_force/rat1.mat')
force = mat['Force_new_pre']
force = -force
lfp = mat['LFP_new_pre']
fs = 1000
num_channel = 16
filter_degree = 4
downsample_rate = int(np.round(fs/10))
lag = 10
num_fold = 10
validation_ratio = 0.2
plot_cont = 0

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

    feature = sig.filtfilt(b, a, lfp, axis=1, padtype='odd', method='pad', padlen=3*(max(len(a), len(b)) - 1))
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

fold_r2_test = np.zeros((1, num_fold))
fold_r_test = np.zeros((1, num_fold))

fold_r2_val = np.zeros((1, num_fold))
fold_r_val = np.zeros((1, num_fold))

fold_r2_train = np.zeros((1, num_fold))
fold_r_train = np.zeros((1, num_fold))

reg_lambda = np.linspace(0, 10000, 100)
lambda_grid_test = np.zeros((len(reg_lambda), num_fold))
lambda_grid_val = np.zeros((len(reg_lambda), num_fold))

for reg_val in range(len(reg_lambda)):
    for fold_count in range(num_fold):

        index_train = kfold_index
        index_test = range(int(fold[fold_count]), int(fold[fold_count + 1]))
        index_train = np.delete(index_train, index_test)
        index_val = index_train[-1-np.int(validation_ratio * index_train.size):-1]
        index_train = np.delete(index_train, index_val)
        # Feature train-test separation
        feature_allband_train = feature_allband[index_train, :]
        feature_allband_val = feature_allband[index_val, :]
        feature_allband_test = feature_allband[index_test, :]
        # force train-test separation
        force_train = force[index_train]
        force_val = force[index_val]
        force_test = force[index_test]
        """"
        model = Sequential()
        model.add(Dense(1, activation='relu',use_bias=True, input_dim=1408,kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam',
                      loss='mean_absolute_error')
    
        # Train the model, iterating on the data in batches of 32 samples
        model.fit(feature_allband_train, force_train, verbose=0, epochs=50, batch_size=None)
        prediction = model.predict(feature_allband_test)
        """
        [prediction_train, prediction_train_no_active,  prediction_val, prediction_val_no_active, prediction_test, prediction_test_no_active, a1, a2, b1, b2] = mlp_mse(feature_allband_train, force_train, feature_allband_val, force_val, feature_allband_test, force_test, reg_lambda[reg_val])
        prediction_test = prediction_test.T
        prediction_val = prediction_val.T
        prediction_train = prediction_train.T
        #MyMLP(feature_allband_train, force_train, feature_allband_test, force_test)
        #pls = PLSRegression(n_components=10)
        #pls.fit(feature_allband_train, force_train)

        #prediction = pls.predict(feature_allband_test)

        # Calculate scores
        print("prediction for train")
        R2_score = 1 - LossR2(force_train, prediction_train)
        Corr = LossCorr(force_train, prediction_train)
        fold_r2_train[0, fold_count] = R2_score
        fold_r_train[0, fold_count] = Corr
        print("=============================")
        print("predictions for Train in fold ", fold_count + 1)
        print("r2 score is ", R2_score)
        print("Corrolation score is ", Corr)
        print("Alpha and beta are ", a1, a2, b1, b2)
        print("++++++++++++++++++++++++++++++++++++")

        # Calculate scores
        print("prediction for Validation")
        R2_score = 1 - LossR2(force_val, prediction_val)
        Corr = LossCorr(force_val, prediction_val)
        fold_r2_val[0, fold_count] = R2_score
        fold_r_val[0, fold_count] = Corr
        print("=============================")
        print("predictions for validation in fold ", fold_count + 1)
        print("r2 score is ", R2_score)
        print("Corrolation score is ", Corr)
        print("Alpha and beta are ", a1, a2, b1, b2)
        print("++++++++++++++++++++++++++++++++++++")
        lambda_grid_val[int(reg_val), fold_count] = R2_score
        np.savetxt("grid_val.csv", lambda_grid_val, delimiter=",")


        print("Prediction for Test")
        R2_score = 1 - LossR2(force_test, prediction_test)
        Corr = LossCorr(force_test, prediction_test)
        fold_r2_test[0, fold_count] = R2_score
        fold_r_test[0, fold_count] = Corr
        print("=============================")
        print("predictions for fold ", fold_count+1)
        print("r2 score is ", R2_score)
        print("Corrolation score is ", Corr)
        print("Alpha and beta are ", a1, a2, b1, b2)

        lambda_grid_test[int(reg_val), fold_count] = R2_score
        np.savetxt("grid_test.csv", lambda_grid_test, delimiter=",")


        if plot_cont == 1:
            softplus_4variable(a1, a2, b1, b2)

            plt.figure()
            t = np.linspace(0, 1, prediction_test_no_active.shape[1])
            plt.plot(t, force_test, t, prediction_test, t, prediction_test_no_active.T, t, -b2*np.ones((t.shape[0], 1)))
            plt.legend(['True Value', 'prediction', 'prediction not activation', 'Low Threshold'])
            plt.show()

            plt.figure()
            t = np.linspace(0, 1, prediction_train_no_active.shape[1])
            plt.plot(t, force_train, t, prediction_train, t, prediction_train_no_active.T, t, -b2*np.ones((t.shape[0], 1)))
            plt.legend(['True Value', 'prediction', 'prediction not activation', 'Low Threshold'])
            plt.show()

    print('===========================================')
    print('===========================================')
    print('========Average for all 10 folds===========')
    print('===========================================')
    print('Regvalue is ', reg_lambda[reg_val], ' Mear r2 Test ', np.mean(fold_r2_test), 'Mean r Test ', np.mean(fold_r_test))
    print('===========================================')
    print('Train Average R2 Value ', fold_r2_train)
    print(' Average ', np.mean(fold_r2_train))
    print('Train Average Corr Value ', fold_r_train)
    print('Average ', np.mean(fold_r_train))
    print('******')
    print('Validation Average R2 Value ', fold_r2_val)
    print('Average ', np.mean(fold_r2_val))
    print('Validation Average Corr Value ', fold_r_val)
    print('Average ', np.mean(fold_r_val))
    print('******')
    print('Test Average R2 Value ', fold_r2_test)
    print('Average ', np.mean(fold_r2_test))
    print('Test Average Corr Value ', fold_r_test)
    print('Average ', np.mean(fold_r_test))



