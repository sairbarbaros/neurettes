
import numpy as np
from neurettes.neuros import compute_cost


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





    
