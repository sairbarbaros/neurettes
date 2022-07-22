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

        parameters['W' + str(l)] = np.random.randn(
            layer_dim[l], layer_dim[l-1]) * np.sqrt(2./layer_dim[l-1])

        parameters['b' + str(l)] = np.zeros((layer_dim[l], 1)) * np.sqrt(2. / layer_dim[l - 1])

    
    return parameters

def sigmoid(Z):
    """Activate your linearly computed output of parameters with sigmoid activation function.

    :param Z: pre-activation parameter
    :type Z: np.array(size of current layer, num of examples)

    :return: The activated parameter A and activation_cache containing A and Z
    :rtype: np.array(size of previous layer, num of examples), tuple(A, Z)
    """
    
    A = 1/(1 + np.exp(-Z))
    activation_cache = (A, Z)

    return A, activation_cache

def sigmoid_derivative(dA, activation_cache):
    """compute the derivative of the activation

    :param dA: gradient of activation
    :type: np.array
    :param activation cache: containing linear and activation cache

    :returns: derivative of Z, dZ
    :rtype: np.array
    """ 

    activation_cache = np.transpose(activation_cache)
    dZ = np.multiply(dA, np.multiply(activation_cache[0][:], (1 - activation_cache[0][:])))

    return dZ

def relu(Z):
    """Activate your linearly computed output of parameters with sigmoid activation function.

    :param Z: pre-activation parameter
    :type Z: np.array(size of current layer, num of examples)

    :return: The activated parameter A and pre-activation parameter Z
    :rtype: np.array(size of previous layer, num of examples)
    """

    A = max(0.0, Z)

    return A, Z

def relu_derivative(dA, activation_cache):
    """compute the derivative of the activation

    :param dA: gradient of activation
    :type: np.array
    :param activation cache: containing linear and activation cache

    :returns: derivative of Z, dZ
    :rtype: np.array
    """ 

    activation_cache = np.transpose(activation_cache)
    dZ = np.multiply(dA, np.greater(activation_cache[1][:] > 0))
    

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

    Z = np.dot(A, W) + b
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

    for l in range(1, L-1):

        prevA = A
        A, cache = activation(prevA, parameters['W' + str(l)],
         parameters['b' + str(l)], "relu")

        caches.append(cache)
    
    prev_A = A
    A_prev, cache = activation(prev_A,parameters['W' + str(L-1)],
         parameters['b' + str(L-1)], "relu")
    caches.append(cache)
    
    lastA, cache = activation(A_prev, parameters['W' + str(L)],
         parameters['b' + str(L)], "sigmoid")
    
    caches.append(cache)

    return lastA, A_prev, caches

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
    cost = (-1/m) * (np.dot(Y, np.transpose(np.log(lastA)) + np.dot((1-Y), np.transpose(np.log(1-lastA)))))
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

    dW = (1/m) * np.dot(dZ, np.transpose(prevA))
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dprevA = np.dot(np.transpose(W), dZ)

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

    current_cache = caches[L-1]
    dlastA = -(np.divide(Y, lastA) - np.divide(1-Y, 1-lastA))
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db', str(L)] = backward_activation(dlastA, current_cache, "sigmoid")

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dprevA_pseudo, dW_pseudo, db_pseudo = backward_activation(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dprevA_pseudo
        grads["dW" + str(l+1)] = dW_pseudo
        grads["db" + str(l+1)] = db_pseudo

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
    L = len(parameters_tbu)//2

    for l in range(L):

        parameters_tbu['W' + str(l+1)] = parameters_tbu['W' + str(l+1)] - learning_rate*grads['dW' + str(l+1)]
        parameters_tbu['b' + str(l+1)] = parameters_tbu['b' + str(l+1)] - learning_rate*grads['db' + str(l+1)]

    return parameters_tbu


def L2_cost(lastA, Y, parameters, lambd):

    """Implement L2 regularization for computing cost

    :param lastA: last activated parameter
    :type lastA: np.array
    :param Y: true labels which can be only 0 or 1
    :type Y: np.array
    :param parameters: dictionary of parameters
    :type parameters: dictionary
    :param lambd: lambda, the constant of L2 regularization
    :type lambd: float

    :return: revised cost
    :rtype: float
    """

    m = Y.shape[1]
    prev_cost = compute_cost(lastA, Y)
    sum_params = 0
    
    for i in range(1, m+1):

        sum_params += np.sum(np.square(parameters['W' + str(i)]))

    L2_to_add = (1/m)*(lambd/2)*sum_params
    cost = prev_cost + L2_to_add

    return cost

def L2_backward_step(dZ, linear_cache, lambd):
    """Implementing L2 regularization for linear-backward step

    :param dZ: derivative of pre-activation parameter
    :type dZ: np.array
    :param linear_cache: containing W, b
    :type: dictionary
    :param lambd: constant of L2 regularization
    :type: float
    :param lambd: constant of L2 regularization
    :type: float

    :return: derivative of prev_A, W, b
    :rtype: np.array
    """

    prev_A, W, b = linear_cache
    m  = prev_A.shape[1]

    dW = (1/m) * np.dot(dZ, np.transpose(prev_A)) + (lambd/m)*W
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    return dA_prev, dW, db

def L2_backward_activation(A, A_prev, Y, cache, lambd, is_it_first):
    """Implementing L2 regularization for backward activation

    :param A: activated parameter
    :type: np.array
    :param cache: containing linear and activation caches
    :type cache: tuple
    :param lambd: L2 regularization constant
    :type lambd: float
    :param is_it_last: is it first computing
    :type: boolean
    
    :return: dA_prev, dW, db
    :rtype: np.array
    """

    linear_cache, activation_cache = cache

    if is_it_first:

        dZ = A - Y
        dA_prev, dW, db = L2_backward_step(dZ, linear_cache, lambd)

    else:

        dZ = np.multiply(dA_prev, np.int64(A_prev > 0))
        dA_prev, dW, db = L2_backward_step(dZ, linear_cache, lambd)

    return dA_prev, dW, db



def L2_backward_model(lastA, A_prev, Y, caches, lambd):
    """Implement L2 regularization for backpropagation
    
    :param lastA: last activated parameter
    :type lastA: np.array
    :param Y: true labels which can only be 0 or 1
    :type Y: np.array
    :param caches: containing linear and activation caches
    :type caches: list
    :param lambd: L2 regularization constant
    :type lambd: float

    :return gradients of parameters
    :rtype: dictionary
    """

    L = len(caches)
    m = lastA.shape[1]
    grads = {}
    Y = Y.reshape(lastA.shape)

    current_cache = caches[L-1]
    dlastA = -(np.divide(Y, lastA) - np.divide(1-Y, 1-lastA))
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db', str(L)] = L2_backward_activation(lastA, A_prev, Y, current_cache, lambd, is_it_first=True)

    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dprevA_pseudo, dW_pseudo, db_pseudo = L2_backward_activation(grads["dA" + str(l + 1)], A_prev, current_cache, lambd, is_it_first=False)
        grads["dA" + str(l)] = dprevA_pseudo
        grads["dW" + str(l+1)] = dW_pseudo
        grads["db" + str(l+1)] = db_pseudo

    return grads



def train_model(X, Y, layer_dim, learning_rate=0.01, iteration=2000, print_cost=True, lambd=0):
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

        lastA, A_prev, caches = forward_model(X, parameters)
        if lambd == 0:
            
            cost = compute_cost(lastA, Y)
            grads = backward_model(lastA, Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)

        elif lambd != 0:

            cost = L2_cost(lastA, Y, parameters)
            grads = L2_backward_model(lastA, A_prev, Y, caches, lambd)
            parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost==True and i % 100==0:
            print(i + "th iteration, cost is" + cost)
            costs.append(cost)

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

    y_hat, _, _ = forward_model(X, parameters)
    m = y_hat.shape[1]
    counter = 0
    subs = np.subtract(y_hat, Y)

    for i in range(0, m):

        if -0.5<= subs[i] <= 0.5:
            counter += 1
        
    accuracy = (counter/m)*100
    
    if print_accuracy:
        print(accuracy)
    
    return accuracy

