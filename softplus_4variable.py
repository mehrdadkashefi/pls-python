# A mixture of two softplus activations.
# Programmer: Mehrdad Kashefi

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
def softplus_4variable(a1, a2, b1, b2):
    t = np.linspace(-10, 10, 1000)
    t = np.reshape(t, (len(t), 1))
    custom_softplus_reverse = -np.log(1+np.exp(a2*(-t - b2))) +np.log(1+np.exp(a1*(t - b1)))
    custom_softplus_reverse_derivative = a1*np.exp(a1*(t - b1))/(1+np.exp(a1*(t - b1))) + a2*np.exp(a2*(-t - b2)) /(1+np.exp(a2*(-t - b2)))
    custom_softplus_reverse_derivative_a1 = t*np.exp(a1*(t - b1))/(1+np.exp(a1*(t - b1)))
    custom_softplus_reverse_derivative_a2 = t*np.exp(a2*(-t - b2))/(1+np.exp(a2*(-t - b2)))
    custom_softplus_reverse_derivative_b1 = -a1*np.exp(a1*(t - b1))/(1+np.exp(a1*(t - b1)))
    custom_softplus_reverse_derivative_b2 = a2*np.exp(a2*(-t - b2))/(1+np.exp(a2*(-t - b2)))


    plt.figure()
    plt.plot(t, custom_softplus_reverse)
    # plt.plot(t, custom_softplus_reverse_derivative)
    # plt.plot(t, custom_softplus_reverse_derivative_a1)
    # plt.plot(t, custom_softplus_reverse_derivative_a2)
    # plt.plot(t, custom_softplus_reverse_derivative_b1)
    # plt.plot(t, custom_softplus_reverse_derivative_b2)

    plt.legend(('Custom Softplus', 'Derivative of Custom Softplus', 'Derivative a1', 'Derivative a2', 'Derivative b1', 'Derivative b2'))
    plt.title('Custom activation')
    plt.grid(True)
    plt.show()
