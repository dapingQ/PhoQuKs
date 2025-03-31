"""Runs the whole set of parameters defined in the lists below."""

from discopy.quantum.optics import ansatz, params_shape
import numpy as np
np.set_printoptions(precision=6)

import multiprocessing as mp
import time
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
from sklearn.model_selection import GridSearchCV, ParameterGrid
import pickle
from geo_diff import *    # Definitions of geometric difference etc from separate module file

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

n_iterations = 5
regularization = 0.02
test_size = 1 / 3

state_conventions = ["Left", "Cent"]    # "Cent" or "Left"
# widths = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
widths = [5, 6, 7]

dps = [40, 60, 80, 100, 120, 140, 160, 180, 200]
# dps = [40]

photon_nums = [2, 3]

experiment_suffix = ""

def get_state(w, n, convention):
    """Returns state of length w with n photons either on the left or centred (or as centred as possible)"""
    if convention == "Cent":    
        state = [0]*w
        start = (w - n)//2
        end = (w + n)//2
        state[start:end] = [1]*n
        
    elif convention == "Left":
        state = [1]*n    # Create state with n_photons on left and rest empty
        state.extend([0]*(w - n))

    return state


# Define grid of parameters to iterate over

hyperparam_grid = {'n_photons': photon_nums,
                    'widths': widths,
                   'dps': dps,
                   'n_photons': photon_nums,
                   'state_conventions': state_conventions}

grid = ParameterGrid(hyperparam_grid)

start = time.time()

def calculate(hyperparams):
    n_photons = hyperparams['n_photons']
    width = hyperparams['widths']
    depth = width
    N = hyperparams['dps']
    state_convention = hyperparams['state_conventions']
    
    
    state = get_state(width, n_photons, state_convention)

    n_params = np.prod(params_shape(width, depth))
   
    def random_datapoint():
       return np.random.uniform(0, 2, size=(n_params,))

    def chip(params):
        return ansatz(width, depth, params)

    def get_time():
        return (time.time()-iteration_start)

    
    q_accs = []
    c_accs = []
    g_accs = []
    l_accs = []
    p_accs = []
    s_accs = []
    Xs= []
    ys = []
    c_grams = []
    q_grams = []

    filename = "../all_kernels_simulations/allkernels_iterations{}state{}w{}d{}n_photons{}N{}_{}.pickle".format(n_iterations, state_convention, width, depth, n_photons, N, experiment_suffix)

    try:    
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            # print(data['N'])
    
    except FileNotFoundError:
        print(filename, '\nNot found')    
        return
    
    for i in range(n_iterations):

        os.system('cls')

        print(f"""Current parameters:
        n_photons: {n_photons}
        depth: {depth}
        width: {width}
        n_params: {N}
        state: {state}""")

        if i > 0:
            print("Previous iteration time: {}".format(time.time() - iteration_start))
        iteration_start = time.time()
        print(f"Iteration {i+1}/{n_iterations}")


        # Generate datapoints
        # X = [random_datapoint() for _ in range(N)]
        X=data['Xs'][i]
        # Define kernels and get gram matrices
        indist_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).indist_prob(state, state)
        indist_gram = gram_matrix(indist_kernel(state), X)
        print("Get indist gram ", get_time())

        c_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).dist_prob(state, state)
        c_gram = gram_matrix(c_kernel(state), X)
        print("Get c kernel ", get_time())

        # Separate according to indist kernel:
        g, y = geometric_diff(c_gram, indist_gram, reg=regularization)
        sQ = model_complexity(indist_gram, y, reg=False)
        sC = model_complexity(c_gram, y, reg=regularization)
        print("sQ = {}, sC = {}, g = {}".format(sQ, sC, g))
        print(sQ, sC*g**2)

        if not np.isclose(sC, sQ*g**2):
            print("Warning: sC != sQ*g**2")
            indist_acc_test = 0
            c_acc_test = 0
            g_acc_test = 0
            
            q_accs.append(indist_acc_test)
            c_accs.append(c_acc_test)
            # g_accs.append(a_gauss_test)

            Xs.append(X)
            ys.append(y) 

        else:

        #     y_labels = [1 if z > np.mean(y) else -1 for z in y]
        #     print("Mean of y is {}, labels sum to {}".format(np.mean(y), sum(y_labels)))

        #     # Define test and train data
        #     num_train = int((1 - test_size) * N)
        #     num_test = N - num_train
        #     indexes = np.random.choice(range(N), num_train, replace=False)

        #     X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(X, y_labels, range(N), test_size=test_size, random_state=RANDOM_STATE)
            

        #     # Get quantum/classical grams
        #     indist_gram_train = np.array([[indist_gram[i, j] for j in indexes_train] for i in indexes_train])
        #     indist_gram_test = np.array([[indist_gram[i, j] for j in indexes_train] for i in indexes_test])

        #     c_gram_train = np.array([[c_gram[i, j] for j in indexes_train] for i in indexes_train])
        #     c_gram_test = np.array([[c_gram[i, j] for j in indexes_train] for i in indexes_test])

        #     # Test quantum/classical kernels
        #     classifier = svm.SVC(kernel='precomputed', verbose=False)
            
        #     classifier.fit(indist_gram_train, y_train)
        #     indist_acc_test = accuracy_score(y_test, classifier.predict(indist_gram_test))
        #     print("\nIndist accuracy test: {}".format(indist_acc_test))
            
        #     classifier.fit(c_gram_train, y_train)
        #     c_acc_test = accuracy_score(y_test, classifier.predict(c_gram_test))
        #     print("C accuracy test: {}".format(c_acc_test))
            
            

            # GAUSSIAN MODEL 
            # reg_param = 0.025
            # gauss = lambda gamma: lambda x0, x1: np.exp(- gamma * norm(x0 - x1) ** 2)
            
            # # Get geometric differences for Gaussian model
            # geos = {}
            # for gamma in np.logspace(-4, 1, num=100):
            #     g_gram = np.array([[gauss(gamma)(xi, xj) for xi in X] for xj in X])
            #     geos.update({gamma : geometric_diff(g_gram, indist_gram, reg=reg_param)[0]})

            # gamma_min = min(geos, key=geos.get)
            
            # gauss_train = np.array([[gauss(gamma_min)(xi, xj) for xi in X_train] for xj in X_train])
            # classifier.fit(gauss_train, y_train)
            # a_gauss_train = accuracy_score(y_train, classifier.predict(gauss_train))
        
            # gauss_test = np.array([[gauss(gamma_min)(xi, xj) for xi in X_train] for xj in X_test])
            # a_gauss_test = accuracy_score(y_test, classifier.predict(gauss_test))
            
            
            # C_range = np.logspace(-2, 3, num=50)
            # gamma_range = np.logspace(-4, 3, num=100)
            # param_grid = dict(gamma=gamma_range, C=C_range)
            # grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5) #, iid=False)
            # grid.fit(X_train, y_train)
            
            # a_g_train = accuracy_score(y_train, grid.predict(X_train))
            # a_g_test = accuracy_score(y_test, grid.predict(X_test))
            
            # # a_gauss_test = max(a_gauss_test, a_g_test)
            # a_gauss_test = a_g_test
            # # Train Gaussian model
            
            # print("G test acc: ", a_gauss_test)
        

        
            # LINEAR MODEL
            # linear = lambda x0, x1: np.inner(x0, x1)

            # l_gram = gram_matrix(linear, X)
            # l_gram_train = np.array([[l_gram[i, j] for j in indexes_train] for i in indexes_train])
            # classifier.fit(l_gram_train, y_train)
            # l_gram_test = np.array([[l_gram[i, j] for j in indexes_train] for i in indexes_test])
            
            # lin_acc_test = accuracy_score(y_test, classifier.predict(l_gram_test))

            # print("Linear accuracy test: {}".format(lin_acc_test))

            # GET POLY MODEL
            # poly = lambda gamma, d, r: lambda x0, x1: (gamma * np.inner(x0, x1) + r) ** d    
            # geos = {}
            # for gamma in np.logspace(-4, 1, num=50):
            #     for d in range(1, 6):
            #         for r in np.logspace(-4, 1, num=50):
            #             p_gram = np.array([[poly(gamma, d, r)(xi, xj) for xi in X] for xj in X])
            #             geos.update({(gamma, d, r) : geometric_diff(p_gram, indist_gram, reg=reg_param)[0]})     

            # gamma_min, d_min, r_min = min(geos, key=geos.get)

            # poly_train = np.array([[poly(gamma_min, d_min, r_min)(xi, xj) for xi in X_train] for xj in X_train])
            # classifier.fit(poly_train, y_train)         
            # poly_test = np.array([[poly(gamma_min, d_min, r_min)(xi, xj) for xi in X_train] for xj in X_test])

            # a_poly_test = accuracy_score(y_test, classifier.predict(poly_test))

            # Use grid search to find best parameters
            # gamma_range = np.logspace(-4, 3, num=50)
            # d_range = range(1, 10)
            # r_range = np.logspace(-4, 1, num=50)
            # param_grid = dict(gamma=gamma_range, degree=d_range, coef0=r_range)
            # grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=5)
            # grid.fit(X_train, y_train)

            # a_p_test = accuracy_score(y_test, grid.predict(X_test))
            # # a_poly_test = max(a_poly_test, a_p_test)
            # a_poly_test = a_p_test
            # print("\n P test acc: ", a_poly_test)

            
        # SIGMOID MODEL
            # sigmoid = lambda gamma, r: lambda x0, x1: np.tanh(gamma * np.inner(x0, x1) + r)
            # geos = {}
            # for gamma in np.logspace(-4, 1, num=100):
            #     for r in np.logspace(-4, 1, num=100):
            #         s_gram = np.array([[sigmoid(gamma, r)(xi, xj) for xi in X] for xj in X])
            #         geos.update({(gamma, r) : geometric_diff(s_gram, indist_gram, reg=reg_param)[0]}) 

            # gamma_min, r_min = min(geos, key=geos.get)

            # sigmoid_train = np.array([[sigmoid(gamma_min, r_min)(xi, xj) for xi in X_train] for xj in X_train])
            # classifier.fit(sigmoid_train, y_train)
            # sigmoid_test = np.array([[sigmoid(gamma_min, r_min)(xi, xj) for xi in X_train] for xj in X_test])

            # a_sigmoid_test = accuracy_score(y_test, classifier.predict(sigmoid_test))

            # Use grid search to find best parameters
            # gamma_range = np.logspace(-4, 3, num=50)
            # r_range = np.logspace(-4, 1, num=50)
            # param_grid = dict(gamma=gamma_range, coef0=r_range)
            # grid = GridSearchCV(svm.SVC(kernel='sigmoid'), param_grid=param_grid, cv=5)
            # grid.fit(X_train, y_train)

            # a_s_test = accuracy_score(y_test, grid.predict(X_test))
            # # a_sigmoid_test = max(a_sigmoid_test, a_s_test)
            # a_sigmoid_test = a_s_test

            # print("\n S test acc: ", a_sigmoid_test)


            # Collect data
            # q_accs.append(indist_acc_test)
            # c_accs.append(c_acc_test)
            # g_accs.append(a_gauss_test)
            # l_accs.append(lin_acc_test)
            # p_accs.append(a_poly_test)
            # s_accs.append(a_sigmoid_test)

            # Xs.append(X)
            # ys.append(y)
            c_grams.append(c_gram)
            q_grams.append(indist_gram)
        
        
    dump_data = dict()
    dump_data.update({
        # 'state': state,
        # 'convention': state_convention,
        # 'testsize': test_size,
        # 'n_iterations': n_iterations,
        # 'width': width,
        # 'depth': depth,
        # 'N': N,
        # 'n_photons': n_photons,
        # 'q_accs': q_accs,
        # 'c_accs': c_accs,
        # 'g_accs': g_accs,
        # 'l_accs': l_accs,
        # 'p_accs': p_accs,
        # 's_accs': s_accs,
        # 'Xs': Xs,
        # 'ys': ys
        'q_grams': q_grams,
        'c_grams': c_grams,
        })
        

    filename = "../all_kernels_simulations/allkernels_iterations{}state{}w{}d{}n_photons{}N{}_{}_new.pickle".format(n_iterations, state_convention, width, depth, n_photons, N, experiment_suffix)
    with open(filename, 'wb') as handle:
        pickle.dump(dump_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    pool = mp.Pool(6)

    with pool as pool:
        pool.map(calculate, grid)
    # for gg in grid:
    #     calculate(gg)