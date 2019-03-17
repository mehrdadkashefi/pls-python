# In this program a a multi-layer perceptron is coded.
# Programmer: Mehrdad Kashefi based on Coursera
# Date: Jan 5 2019

# Importing Libraries
from MLP_Funtions import *
import numpy as np
import matplotlib.pyplot as plt


def mlp_mse(x_train, y_train, x_test, y_test, reg_val):
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Transposing Data to standard NN shape Matrix are in form of (Features x num_samples)
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    num_layer = 1
    num_neuron = np.array([x_train.shape[0], y_train.shape[0]])
    learning_rate = 0.01

    learning_rate_activation_a1 = 0
    learning_rate_activation_a2 = 0
    learning_rate_activation_b1 = 20
    learning_rate_activation_b2 = 20

    random_initializer = 0.01
    regularization_rate = reg_val  # 800 5000
    num_iteration = 500       # 500

    a1 = 1
    a2 = 1
    b1 = 0
    b2 = 0

    [weight, bias, dW, dB] = layer_initializer(num_layer, num_neuron, random_initializer)
    cost = np.zeros((1, num_iteration))
    cost_validation = np.zeros((1, num_iteration))

    # Initializing with pseudo inverse
    x_init = x_train.T
    x_train_init = np.concatenate((np.ones((x_init.shape[0], 1)), x_init), 1)
    I = np.eye(x_train_init.shape[1])
    I[:, 0] = 0
    weight_init = np.dot(np.dot(np.linalg.pinv(np.dot(x_train_init.T, x_train_init) + regularization_rate * I),
                         x_train_init.T), y_train.T)
    weight['w1'] = weight_init[1:weight_init.shape[0], 0:1].T
    bias['b1'] = weight_init[0:1, 0:1]
    # Main Loop of the program
    for iteration in range(num_iteration):

        Z = {}
        output = {'a0' : x_train}

        [output['a1'], Z['z1']] = forward_block(output['a0'], weight['w1'], bias['b1'], activation='my_activation',
                                                a1=a1, a2=a2, b1=b1, b2=b2)
        # Bach Prob
        dA = {}

        dA['da1'] = output['a1'] - y_train

        [dW['dw1'], dB['db1'], dA['da0'], d_a1, d_a2, d_b1, d_b2] = backward_block(dA['da1'],
                                                                                   Z['z1'], weight['w1'], output['a0'],
                                                                                   activation='my_activation', a1=a1,
                                                                                   a2=a2, b1=b1, b2=b2)

        # Updating Weights and Biases

        weight['w1'] = weight['w1'] - learning_rate * (dW['dw1'] + (regularization_rate/x_train.shape[1]) * weight['w1'])

        bias['b1'] = bias['b1'] - learning_rate * dB['db1']

        a1 = a1 - (learning_rate_activation_a1 * d_a1)

        a2 = a2 - (learning_rate_activation_a2 * d_a2)

        b1 = b1 - (learning_rate_activation_b1 * d_b1)

        b2 = b2 - (learning_rate_activation_b2 * d_b2)

        cost[0, iteration] = cost_function(output['a1'], y_train)
        # print("In Iteration ", iteration, " The cost is ", cost[0, iteration])

        # Validation on Test Set

        output['a0'] = x_test

        [output['a1'], Z['z1']] = forward_block(output['a0'], weight['w1'], bias['b1'], activation='my_activation',
                                                a1=a1, a2=a2, b1=b1, b2=b2)

        cost_validation[0, iteration] = cost_function(output['a1'], y_test)
        # print("In Iteration ", iteration, " The test cost is ", cost_validation[0, iteration])

    [output_train, Z_train] = forward_block(x_train, weight['w1'], bias['b1'], activation='my_activation', a1=a1,
                                            a2=a2, b1=b1, b2=b2)
    [output_test, Z_test] = forward_block(x_test, weight['w1'], bias['b1'], activation='my_activation', a1=a1,
                                          a2=a2, b1=b1, b2=b2)
    """"
    fig, ax = plt.subplots()
    ax.plot(range(0, iteration), cost[0, 0:iteration], label='Train Cost')
    ax.plot(range(0, iteration), cost_validation[0, 0:iteration], label='Test Cost')
    ax.set(xlabel='Iterations (#)', ylabel='J cost',
           title='The Learning Curve')
    ax.legend()
    ax.grid(False)
    plt.show()
   """
    return output_train, Z_train, output_test, Z_test, a1, a2, b1, b2
