import numpy as np


def sigmoid(z):
    Sig = 1/(1+np.exp(-z))
    return Sig


def sigmoid_backward(z):
    sig_back = sigmoid(z) * (1-sigmoid(z))
    return sig_back


def relu(z):
    z[z <= 0] = 0
    return z


def relu_backward(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def linear(z):
    return z


def linear_backward(z):
    return 1


def my_activation(z, a1, a2, b1, b2):
    z = -np.log(1+np.exp(a2*(-z - b2))) + np.log(1+np.exp(a1*(z - b1)))
    return z


def my_activation_backward(z, a1, a2, b1, b2):
    z = ((a1*np.exp(a1*(z - b1)))/(1+np.exp(a1*(z - b1)))) + (a2*np.exp(a2*(-z - b2)) / (1+np.exp(a2*(-z - b2))))
    a1 = (z * np.exp(a1 * (z - b1))) / (1 + np.exp(a1 * (z - b1)))
    a2 = (z * np.exp(a2 * (-z - b2))) / (1 + np.exp(a2 * (-z - b2)))
    b1 = (-a1 * np.exp(a1 * (z - b1))) / (1 + np.exp(a1 * (z - b1)))
    b2 = (a2 * np.exp(a2 * (-z - b2))) / (1 + np.exp(a2 * (-z - b2)))
    return z, a1, a2, b1, b2


def layer_initializer(num_layer, num_neuron, random_initializer):

    weight = {}
    bias = {}
    d_weight = {}
    d_bias = {}

    for i in range(num_layer):
        weight["w"+str(i+1)] = np.random.randn(num_neuron[i+1], num_neuron[i]) * np.sqrt(2/num_neuron[i])
        bias["b" + str(i + 1)] = np.random.randn(num_neuron[i + 1], 1) * np.sqrt(2/num_neuron[i])
        d_weight["dw" + str(i + 1)] = np.zeros((num_neuron[i+1], num_neuron[i]))
        d_bias["db" + str(i + 1)] = np.zeros((num_neuron[i+1], 1))

    return weight, bias, d_weight, d_bias


def forward_block(a_in, w, b, activation, a1, a2, b1, b2):
    z = np.dot(w, a_in) + b

    if activation == "sigmoid":
        a_out = sigmoid(z)
        return a_out, z

    elif activation == "relu":
        a_out = relu(z)
        return a_out, z

    elif activation == "linear":
        a_out = linear(z)
        return a_out, z

    elif activation == 'my_activation':
        a_out = my_activation(z, a1, a2, b1, b2)
        return a_out, z


def cost_function(y_prediction, y_true):
    # loss = -1*(y_true * np.log(y_prediction) + (1 - y_true) * np.log((1 - y_prediction)))
    loss = 0.5*((y_true-y_prediction)**2)
    cost = (1/y_prediction.shape[1]) * np.sum(loss, axis=1, keepdims=True)
    return cost


def backward_block(da, z, w, a_prev, activation, a1, a2, b1, b2):
    if activation == 'sigmoid':
        dz = da * sigmoid_backward(z)
    elif activation == "linear":
        dz = da * linear_backward(z)
    elif activation == 'relu':
        dz = da * relu_backward(z)
    elif activation == 'my_activation':
        [z,  a1, a2, b1, b2] = my_activation_backward(z, a1, a2, b1, b2)
        dz = da * z
        d_a1 = da * z * a1
        d_a2 = da * z * a2
        d_b1 = da * z * b1
        d_b2 = da * z * b2

        d_a1 = (1 / dz.shape[1]) * np.sum(d_a1, axis=1, keepdims=True)
        d_a2 = (1 / dz.shape[1]) * np.sum(d_a2, axis=1, keepdims=True)
        d_b1 = (1 / dz.shape[1]) * np.sum(d_b1, axis=1, keepdims=True)
        d_b2 = (1 / dz.shape[1]) * np.sum(d_b2, axis=1, keepdims=True)

    dw = (1 / dz.shape[1]) * np.dot(dz, a_prev.T)
    db = (1 / dz.shape[1]) * np.sum(dz, axis=1, keepdims=True)

    da_prev = np.dot(w.T, dz)

    return dw, db, da_prev, d_a1, d_a2, d_b1, d_b2


def dropout(dropout_input, dropout_prob):
    keep_prob = 1 - dropout_prob
    remover = np.random.rand(dropout_input.shape[0], dropout_input.shape[1]) < keep_prob
    output = np.multiply(remover, dropout_input)
    output /= keep_prob  # keep the mean value of input
    return output
