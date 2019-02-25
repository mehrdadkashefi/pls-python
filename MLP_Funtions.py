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


def my_activation(z, alpha, shift):
    z = -np.log(1 + np.exp(-alpha * z - shift)) + np.log(1 + np.exp(alpha * z - shift))
    return z


def my_activation_backward(z, alpha, shift):
    z = alpha*np.exp(alpha*z - shift)/(1+np.exp(alpha*z - shift)) + alpha*np.exp(-alpha*z - shift)/(1+np.exp(-alpha*z - shift))
    a = z*np.exp(alpha*z - shift)/(1+np.exp(alpha*z - shift)) + z*np.exp(-alpha*z - shift)/(1+np.exp(-alpha*z - shift))
    b = -1*np.exp(alpha*z - shift)/(1+np.exp(alpha*z - shift)) + np.exp(-alpha*z - shift)/(1+np.exp(-alpha*z - shift))
    return z, a, b


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


def forward_block(a_in, w, b, activation, alpha, beta):
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
        a_out = my_activation(z, alpha, beta)
        return a_out, z


def cost_function(y_prediction, y_true):
    # loss = -1*(y_true * np.log(y_prediction) + (1 - y_true) * np.log((1 - y_prediction)))
    loss = 0.5*((y_true-y_prediction)**2)
    cost = (1/y_prediction.shape[1]) * np.sum(loss, axis=1, keepdims=True)
    return cost


def backward_block(da, z, w, a_prev, activation, alpha, beta):
    if activation == 'sigmoid':
        dz = da * sigmoid_backward(z)
    elif activation == "linear":
        dz = da * linear_backward(z)
    elif activation == 'relu':
        dz = da * relu_backward(z)
    elif activation == 'my_activation':
        [z, a, b] = my_activation_backward(z, alpha, beta)
        dz = da * z
        d_alpha = da * z * a
        d_beta = da * z * b

    dw = (1 / dz.shape[1]) * np.dot(dz, a_prev.T)
    db = (1 / dz.shape[1]) * np.sum(dz, axis=1, keepdims=True)
    d_alpha = (1 / dz.shape[1]) * np.sum(d_alpha, axis=1, keepdims=True)
    d_beta = (1 / dz.shape[1]) * np.sum(d_beta, axis=1, keepdims=True)
    da_prev = np.dot(w.T, dz)

    return dw, db, da_prev, d_alpha, d_beta


def dropout(dropout_input, dropout_prob):
    keep_prob = 1 - dropout_prob
    remover = np.random.rand(dropout_input.shape[0], dropout_input.shape[1]) < keep_prob
    output = np.multiply(remover, dropout_input)
    output /= keep_prob  # keep the mean value of input
    return output
