import numpy as np
from scipy.linalg import inv, eigh
from numpy.linalg import norm

def gram_matrix(kernel, x):
    """Build a symmetric positive definite matrix"""
    N = len(x)
    gram = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            gram[i, j] = kernel(x[i], x[j])
    return gram + np.diag(np.ones(N)) + np.transpose(gram)


def geometric_diff(k0, k1, reg=0.0):
    """Calculate geometric difference between two kernels"""
    S0, P0 = eigh(k0)
    S1, P1 = eigh(k1)
    sqrtk0 = P0.dot(np.diag(np.sqrt(np.absolute(S0)))).dot(np.transpose(P0))
    sqrtk1 = P1.dot(np.diag(np.sqrt(np.absolute(S1)))).dot(np.transpose(P1))
    center = P0.dot(np.diag((S0 + reg) ** -2)).dot(np.transpose(P0))
    matrix = sqrtk1.dot(sqrtk0).dot(center).dot(sqrtk0).dot(sqrtk1)
    S, V = eigh(matrix)
    all_eigs_positive = np.all(S > 0)
    index = np.argmax(np.absolute(S))
    return np.sqrt(np.absolute(S[index])), sqrtk1.dot(V[:, index])#, all_eigs_positive


def model_complexity(k, y, reg=False):
    """Calculate model complexity given a kernel and the appropriate eigenvector"""
    if not reg:
        return np.transpose(y).dot(inv(k)).dot(y)
    else:
        S, P = eigh(k)
        sqrtk = P.dot(np.diag(np.sqrt(np.absolute(S)))).dot(np.transpose(P))
        matrix = sqrtk.dot(P).dot(np.diag((S + reg) ** -2)).dot(np.transpose(P)).dot(sqrtk)
        return np.transpose(y).dot(matrix).dot(y)


def get_q_kernel(chip, state, X):
    """Calculate quantum kernel/gram matrix"""
    q_state = state
    q_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).indist_prob(state, state)
    q_gram = gram_matrix(q_kernel(q_state), X)
    return q_kernel, q_gram
    

def get_c_kernel(chip, state, X):
    """Calculate classical kernel/gram matrix"""
    c_state = state
    c_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).dist_prob(state, state)
    c_gram = gram_matrix(c_kernel(c_state), X)
    return c_kernel, c_gram


def get_g_kernel(gamma, X):
    """Calculate Gaussian kernel"""
    # reg_param = 0.025

    g_kernel = lambda gamma: lambda x0, x1: np.exp(- gamma * norm(x0 - x1) ** 2)

    g_gram = np.array([[g_kernel(gamma)(xi, xj) for xi in X] for xj in X])

    # geos = {}
    # for gamma in np.logspace(-4, 1, num=100):
    #     g_gram = np.array([[gauss(gamma)(xi, xj) for xi in X] for xj in X])
    #     geos.update({gamma : geometric_diff(g_gram, q_gram, reg=reg_param)[0]})

    # gamma_min = min(geos, key=geos.get)
    return g_kernel, g_gram
