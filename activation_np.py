import numpy as np


def sigmoid(x):
    """sigmoid
    TODO: 
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    #[TODO 1.1]
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    return a*(1-a)


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    return np.max(0,x) # not checked


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    grad = np.where(a>=0,1,0)
    return grad


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    return None


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    return None


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input size: data demension x num of class // x <=> z
    """
    # each row <-> a sample
    exp_x = np.exp(x)
    sum_row = np.sum(exp_x, axis = 1, keepdims = True) #keepdims = true: sau khi sum xong num of dim not change 
    for i in range(len(sum_row)):      
        exp_x[i,:] = exp_x[i,:]/sum_row[i]
    return exp_x


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """
    x = x.astype(float)  # float
    max_row = np.max(x, axis = 1)
    for i in range(len(max_row)):      
        x[i,:] = np.exp( x[i,:] - max_row[i])
        x[i,:] = x[i,:]/np.sum(x[i,:])
    return x
