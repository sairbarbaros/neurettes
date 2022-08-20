
import numpy as np

def initialize_parameters(layer_dim):
    """Initialize the parameters with He algorithm.

    :param layer_dim: the array of the numbers of each units in the layers
    :type layer_dim: np.array or list

    :return: the dictionary of initialized parameters of all units in the neural network
    :rtype: dictionary

    """

    parameters = {}
    layer_length = len(layer_dim)

    for l in range(1, layer_length):
        parameters['W' + str(l)] = np.random.randn(layer_dim[l], layer_dim[l - 1]) * np.sqrt(2. / layer_dim[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dim[l], 1)) * np.sqrt(2. / layer_dim[l - 1])

    assert (parameters['W' + str(l)].shape == (layer_dim[l], layer_dim[l - 1]))
    assert (parameters['b' + str(l)].shape == (layer_dim[l], 1))

    return parameters


def sigmoid(Z):
    """Activate your linearly computed output of parameters with sigmoid activation function.

    :param Z: pre-activation parameter
    :type Z: np.array(size of current layer, num of examples)

    :return: The activated parameter A and activation_cache containing Z
    :rtype: np.array(size of previous layer, num of examples), Z
    """

    A = 1 / (1 + np.exp(-Z))
    activation_cache = Z

    return A, activation_cache


def sigmoid_derivative(dA, activation_cache):
    """compute the derivative of the activation

    :param dA: gradient of activation
    :type: np.array
    :param activation cache: containing linear and activation cache

    :returns: derivative of Z, dZ
    :rtype: np.array
    """
    sigmo = 1 / (1 + np.exp(-activation_cache))
    dZ = dA * (1 - sigmo) * (sigmo)
    assert (dZ.shape == activation_cache.shape)

    return dZ


def relu(Z):
    """Activate your linearly computed output of parameters with sigmoid activation function.

    :param Z: pre-activation parameter
    :type Z: np.array(size of current layer, num of examples)

    :return: The activated parameter A and pre-activation parameter Z
    :rtype: np.array(size of previous layer, num of examples)
    """

    A = np.maximum(Z, 0)
    assert (A.shape == Z.shape)

    return A, Z


def relu_derivative(dA, activation_cache):
    """compute the derivative of the activation

    :param dA: gradient of activation
    :type: np.array
    :param activation cache: containing linear and activation cache

    :returns: derivative of Z, dZ
    :rtype: np.array
    """

    dZ = np.array(dA, copy=True)
    dZ[activation_cache <= 0] = 0

    assert (dZ.shape == activation_cache.shape)

    return dZ


def linear_step(A, W, b):
    """Take the linear step of the forward propagation

    :param A: activated parameter
    :type A: np.array(size of previous layer, num of examples)
    :param W: weigths matrix
    :type W: np.array(size of current layer, size of previous layer)
    :param b: bias vector
    :type b: np.array(size of current layer, 1)

    :return: pre-activation parameter Z, linear cache containing A, W, b
    :rtype: np.array(size of current layer), tuple(A, W, b)
    """

    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)

    return Z, linear_cache


def activation(prevA, W, b, activation):
    """Activate your linearly computed input

    :param prevA: activated inputs for linear computation
    :type prevA: np.array(size of previous layer, num of examples)
    :param W: weigths matrix
    :type W: np.array(size of current layer, size of previous layer)
    :param b: bias vector
    :type b: np.array(size of current layer, 1)
    :param activation: activation function
    :type activation: string

    :return: output of activation A, cache containing linear and activation caches
    :rtype: np.array(size of current layer, num of examples), tuple(linear_cache, activaitons_cache)
    """

    if activation == "sigmoid":

        Z, linear_cache = linear_step(prevA, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_step(prevA, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def forward_model(X, parameters):
    """forward propagation for l-layered neural network

    :param X: input data
    :type X: np.array(input size, num of examples)
    :param parameters: output of initialization
    :type parameters: dictionary

    :return: last activated output lastA, A_prev caches containing linear and activation caches
    :rtype: np.array, list
    """

    A = X
    L = len(parameters) // 2
    caches = []

    for l in range(1, L):
        prevA = A
        A, cache = activation(prevA, parameters['W' + str(l)],
                              parameters['b' + str(l)], "relu")

        caches.append(cache)

    lastA, cache = activation(A, parameters['W' + str(L)],
                              parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return lastA, caches


def compute_cost(lastA, Y):
    """Compute cost to find gradients

    :param lastA: predictions ranging from 0 and 1
    :type lastA: np.array(1, num of examples)
    :param Y: true labels which can be only 0 or 1
    :type Y: np.array(1, num of examples)

    :return: cross-entropy cost
    :rtype: float32
    """

    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(Y, np.log(lastA).T) + np.dot((1 - Y), np.log(1 - lastA).T))
    cost = np.squeeze(cost)

    return cost


def backward_step(dZ, linear_cache):
    """Take the linear step of the backward propagation

    :param dZ: gradient of the linear output
    :type dZ: np.array
    :param linear_cache: tuple of values from forward propagation

    :return: gradient of activated parameter dprevA, gradient of parameters dW and db
    :rtype: np.array
    """

    prevA, W, b = linear_cache
    m = prevA.shape[1]

    dW = (1 / m) * np.dot(dZ, prevA.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dprevA = np.dot(W.T, dZ)

    return dprevA, dW, db


def backward_activation(dA, cache, activation):
    """Taking a backward step in activation computation

    :param dA: gradient of activated parameter
    :type dA: np.array
    :param cache: containing linear and activation caches
    :type cache: tuple
    :param activation: backward activation function
    :type activation: string

    :return: gradient of previous activated parameter dprevA, gradients of parameters W and b, dW and db
    :rtype: np.array
    """

    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_derivative(dA, activation_cache)
        dprevA, dW, db = backward_step(dZ, linear_cache)

    elif activation == "sigmoid":

        dZ = sigmoid_derivative(dA, activation_cache)
        dprevA, dW, db = backward_step(dZ, linear_cache)

    return dprevA, dW, db


def backward_model(lastA, Y, caches):
    """backward propagation for l-layered neural network

    :param lastA: last activated output
    :type lastA: np.arrray
    :param Y: true labels which can be only 0 or 1
    :type Y: np.array
    :param caches: list of caches containing linear and activation caches
    :type caches: list

    :return: dictionary of gradients
    :rtype: dictionary
    """

    L = len(caches)
    m = lastA.shape[1]
    grads = {}
    Y = Y.reshape(lastA.shape)

    current_cache = caches[L - 1]
    dlastA = -(np.divide(Y, lastA) - np.divide(1 - Y, 1 - lastA))
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db', str(L)] = backward_activation(dlastA, current_cache,
                                                                                              "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dprevA_pseudo, dW_pseudo, db_pseudo = backward_activation(grads["dA" + str(l + 1)], current_cache,
                                                                  activation="relu")
        grads["dA" + str(l)] = dprevA_pseudo
        grads["dW" + str(l + 1)] = dW_pseudo
        grads["db" + str(l + 1)] = db_pseudo

    return grads


def update_parameters(parameters, grads, learning_rate):
    """Update parameters with gradient descent algorithm

    :param parameters: dictionary of parameters
    :type parameters: dictionary
    :param grads: dictionary of gradients
    :type grads: dictionary

    :return: dictionary containing updated parameters
    :rtype: dictionary
    """

    parameters_tbu = parameters.copy()
    L = len(parameters_tbu) // 2

    for l in range(1,L-1):
        parameters_tbu['W' + str(l+1 )] = parameters_tbu['W' + str(l +1)] - learning_rate * grads['dW' + str(l+1)]
        parameters_tbu['b' + str(l +1)] = parameters_tbu['b' + str(l+1 )] - learning_rate * grads['db' + str(l +1)]

    return parameters_tbu


def train_model(X, Y, layer_dim, learning_rate=0.01, iteration=2000, print_cost=True):
    """train your model with forward and backward propagations using gradient descent
    :param X: input data
    :type X: np.array(input size, num of examples)
    :param Y: true labels which can be only 0 or 1
    :type Y: np.array(input size, num of examples)
    :param layer_dim: size of layers
    :type layer_dim: np.array or list
    :param learning_rate: how fast will gradient descent be implemented?
    :type learning_rate: float
    :param iteration: how many times will gradient descent iterate itself?
    :type iteration: integer
    :param print_cost: True if user wants to print cost
    :type print_cost: boolean

    :return: trained parameters, list of costs
    :rtype: dictionary, list
    """

    parameters = initialize_parameters(layer_dim)
    costs = []

    for i in range(0, iteration):

        lastA, caches = forward_model(X, parameters)
        cost = compute_cost(lastA, Y)
        grads = backward_model(lastA, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost == True and i % 100 == 0:
            print("Cost {}: {}, Accuracy: {}".format(i, np.squeeze(cost), (1-cost)*100))
            costs.append(cost)


    return parameters, costs


def predict(parameters, X, Y, print_accuracy=False):
    """test how accurate your system is

    :param X: input data
    :type X: np.array
    :param Y: true labels
    :type Y: np.array

    :return: accuracy
    :rtype: float
    """

    y_hat, _= forward_model(X, parameters)
    m = y_hat.shape[1]
    counter = 0
    subs = np.subtract(y_hat, Y)

    for i in range(0, m):

        if -0.5 <= subs[i] <= 0.5:
            counter += 1

    accuracy = (counter / m) * 100

    if print_accuracy:
        print(accuracy)

    return accuracy
