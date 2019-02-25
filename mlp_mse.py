# In this program a a multi-layer perceptron is coded.
# Programmer: Mehrdad Kashefi based on Coursera
# Date: Jan 5 2019

# Importing Libraries
from MLP_Funtions import *
import numpy as np
import matplotlib.pyplot as plt


def mlp_mse(x_train, y_train, x_test, y_test):
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Transposing Data to standard NN shape Matrix are in form of (Features x num_samples)
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    num_layer = 1
    num_neuron = np.array([x_train.shape[0], y_train.shape[0]])
    learning_rate = 0.001
    random_initializer = 0.01
    regularization_rate = 0.01
    num_iteration = 200

    [weight, bias, dW, dB] = layer_initializer(num_layer, num_neuron, random_initializer)
    cost = np.zeros((1, num_iteration))
    cost_validation = np.zeros((1, num_iteration))

    # Main Loop of the program
    for iteration in range(num_iteration):

        Z = {}
        output = {'a0' : x_train}

        [output['a1'], Z['z1']] = forward_block(output['a0'], weight['w1'], bias['b1'], activation='my_activation')
        # Bach Prob
        dA = {}

        dA['da1'] = output['a1'] - y_train

        [dW['dw1'], dB['db1'], dA['da0']] = backward_block(dA['da1'], Z['z1'], weight['w1'], output['a0'], activation='my_activation')

        # Updating Weights and Biases

        weight['w1'] = weight['w1'] - learning_rate * (dW['dw1'] + (regularization_rate/x_train.shape[1]) * weight['w1'])

        bias['b1'] = bias['b1'] - learning_rate * dB['db1']

        cost[0, iteration] = cost_function(output['a1'], y_train)
        print("In Iteration ", iteration, " The cost is ", cost[0, iteration])

        # Validation on Test Set

        output['a0'] = x_test

        [output['a1'], Z['z1']] = forward_block(output['a0'], weight['w1'], bias['b1'], activation='my_activation')

        cost_validation[0, iteration] = cost_function(output['a1'], y_test)
        print("In Iteration ", iteration, " The test cost is ", cost_validation[0, iteration])

    fig, ax = plt.subplots()
    ax.plot(range(0, iteration), cost[0, 0:iteration], label='Train Cost')
    ax.plot(range(0, iteration), cost_validation[0, 0:iteration], label='Test Cost')
    ax.set(xlabel='Iterations (#)', ylabel='J cost',
           title='The Learning Curve')
    ax.legend()
    ax.grid(False)
    plt.show()

    return output['a1']
