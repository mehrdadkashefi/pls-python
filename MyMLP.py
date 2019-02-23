# In this program a a multi-layer perceptron is coded.
# Programmer: Mehrdad Kashefi based on Coursera
# Date: Jan 5 2019

# Importing Libraries
from MLP_Funtions import layer_initializer, forward_block
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def cost_function(y_prediction, y_true):
    cost = 0.5*((y_prediction-y_true)**2)
    return cost

def MyMLP(x_train, y_train, x_test, y_test):

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    # Transposing Data to standard NN shape Matrix are in form of (Features x num_samples)
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    num_layer = 1
    num_neuron = np.array([x_train.shape[0], y_train.shape[0]])
    random_initializer = 0.01
    learning_rate = 0.001
    num_iteration = 100

    [weight, bias, dW, dB] = layer_initializer(num_layer, num_neuron, random_initializer)
    cost = np.zeros((x_train.shape[1], num_iteration))
    cost_iteration = np.zeros((1, num_iteration))
    cost_validation = np.zeros((x_test.shape[1], num_iteration))
    cost_validation_iteration = np.zeros((1, num_iteration))

    # Main Loop of the program
    for iteration in range(num_iteration):
        for sample in range(x_train.shape[1]):
            Z = {}
            output = {'a0' : x_train[:, sample]}

            [output['a1'], Z['z1']] = forward_block(output['a0'], weight['w1'], bias['b1'], activation='linear')
            error = output['a1']-y_train[:, sample]
            dW['dw1'] = error * 1 * x_train[:, sample]
            dB['db1'] = error * 1

            # Updating Weights and Biases

            weight['w1'] = weight['w1'] - learning_rate * dW['dw1']

            bias['b1'] = bias['b1'] - learning_rate * dB['db1']

            cost[sample, iteration] = cost_function(output['a1'], y_train[:, sample])

        # Validation on Test Set
        for test_sample in range(x_test.shape[1]):
            output['a0'] = x_test[:, test_sample]

            [output['a1'], Z['z1']] = forward_block(output['a0'], weight['w1'], bias['b1'], activation='linear')

            cost_validation[test_sample, iteration] = cost_function(output['a1'], y_test[:, test_sample])

        print("In Iteration ", iteration, " The Train cost is ", np.mean(cost[:, iteration], axis=0))
        print("In Iteration ", iteration, " The test cost is ", np.mean(cost_validation[:, iteration], axis=0))
        cost_iteration[:, iteration] = np.mean(cost[:, iteration], axis=0)
        cost_validation_iteration[:, iteration] = np.mean(cost_validation[:, iteration], axis=0)



    fig, ax = plt.subplots()
    ax.plot(range(0, iteration), cost_iteration[0, 0:iteration], label='Train Cost')
    ax.plot(range(0, iteration), cost_validation_iteration[0, 0:iteration], label='Test Cost')
    ax.set(xlabel='Iterations (#)', ylabel='J cost',
           title='The Learning Curve')
    ax.legend()
    ax.grid(False)
    plt.show()

