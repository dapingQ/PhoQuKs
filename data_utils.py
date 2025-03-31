#%% basic functions
from cProfile import label
import time, socket, os, pathlib, sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm, inv, eigh
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from cycler import cycler

cl = ['#0055ff', '#ff5500', 'grey', 'orange', 'violet']
plt.rcParams.update({    
    'axes.prop_cycle': cycler('color', cl),
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'lines.markersize': 4.0,
    'font.size':8,
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': 'Arial',
    'mathtext.fontset': 'cm',
    'legend.framealpha': 1,
    'legend.frameon': False,
    'legend.edgecolor': 'inherit',
    'savefig.transparent': True,
    'savefig.dpi': 400,
    'image.cmap': 'Spectral',#'seismic',
    'pdf.fonttype': 42,
})

# ensure the discopy version a0aeb6c0471440250b06807802e7e94c748b4035
from discopy.quantum.optics import ansatz, params_shape, to_matrix
from discopy.quantum.optics import npperm

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import time, pickle

# data types for counting
count_dt = np.dtype([
    ('sc', 'i', (6,)),
    ('cc', 'i', (15,)),
    ('t', 'f', (1,))
])

# data types for measurement
data_dt = np.dtype([
    ('idx', '<U16'),
    ('counts', [
        ('sc', '<i4', (6,)),
        ('cc', '<i4', (15,)),
        ('t', '<f4', (1,))], (5,))
])

width=6
depth=6

# state used
left_label = 'Left state \n'+r'$\vert 1,1,0,0,0,0 \rangle$'+'\n test accuracy'
cent_label = 'Cent state \n'+r'$\vert 0,0,1,1,0,0 \rangle$'+'\n test accuracy'

# simulation Ns
Ns_all = range(40,220,20) 
# experimental Ns
Ns = range(40,120,20)

test_size = 1 / 3
RANDOM_STATE = 42

def chip(params): return ansatz(width, depth, params)
def sampler(x0, x1): return (chip(x0) >> chip(x1).dagger())

c_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).dist_prob(state, state)
q_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).indist_prob(state, state)
g_kernel = lambda gamma: lambda x0, x1: np.exp(- gamma * norm(x0 - x1) ** 2)

def gram_matrix(kernel, x):
    """
    Build a symmetric positive definite matrix
    """
    N = len(x)
    gram = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            gram[i, j] = kernel(x[i], x[j])
    return gram + np.diag(np.ones(N)) + np.transpose(gram)

def gram_matrix_from_result(result, exp_theo, nn, i):
    '''
    experimental data only take K(x_i, x_j) since K(x_j, x_i) = K(x_i, x_j)
    here to construct the full gram matrix, adding the diagonal and the transpose
    '''
    gram = result[exp_theo][nn,:,:,i]
    return gram + np.diag(np.ones(N)) + np.transpose(gram)

def plot_bar(xdata, xcord=None, xtick=None, *args):
    '''
    plot bar chart
    '''
    if xcord == None:
        xcord = np.arange(len(xdata[0]))
    if xtick == None:
        xtick = [str(i) for i in range(len(xcord))]
    width = 1/(len(xdata)+1)
    fig, ax = plt.subplots(*args)
    xnum = len(xdata)
    for i in range(xnum):
        ax.bar(xcord+(i-xnum/2-0.5)*width, xdata[i], width)
    ax.set_xticks(xcord, xtick)
    return fig, ax

def geometric_diff(k0, k1, reg=0.0):
    """
    Calculate geometric difference between two kernels
    k0, k1: kernel matrices
    reg: regularization parameter
    """
    S0, P0 = eigh(k0)
    S1, P1 = eigh(k1)
    sqrtk0 = P0.dot(np.diag(np.sqrt(np.absolute(S0)))).dot(np.transpose(P0))
    sqrtk1 = P1.dot(np.diag(np.sqrt(np.absolute(S1)))).dot(np.transpose(P1))
    center = P0.dot(np.diag((S0 + reg) ** -2)).dot(np.transpose(P0))
    matrix = sqrtk1.dot(sqrtk0).dot(center).dot(sqrtk0).dot(sqrtk1)
    S, V = eigh(matrix)
    all_eigs_positive = np.all(S > 0)
    index = np.argmax(np.absolute(S))
    return np.sqrt(np.absolute(S[index])), sqrtk1.dot(V[:, index])

def get_key_list(dicts, key, shape_pre = (4,2)):
    nd = np.array([ list(d[key]) for d in dicts.values()])
    nd.shape = shape_pre + nd.shape[1:]
    return nd
    
def gram_classifier(X, y, gram, N):
    '''
    support vector machine classifier with precomputed kernel
    X: data
    y: labels
    gram: kernel matrix
    N: dataset size
    '''
    
    y_labels = [1 if z > np.mean(y) else -1 for z in y]
    X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(X, y_labels, range(N), test_size=test_size, random_state=RANDOM_STATE)

    gram_train = np.array([[gram[i, j] for j in indexes_train] for i in indexes_train])
    gram_test = np.array([[gram[i, j] for j in indexes_train] for i in indexes_test])
    
    # Test quantum/classical kernels
    classifier = svm.SVC(kernel='precomputed', verbose=False)  
    try:
        classifier.fit(gram_train, y_train)
        acc_test = accuracy_score(y_test, classifier.predict(gram_test))
    except:
        print(N, nn)
    return acc_test


def p2s(pat, width):
    '''
    Convert pattern number to state.
    eg [3,4] width 6 <=> [0,0,1,1,0,0]
    '''
    assert max(pat) <= width
    # return [1 if i in pat else 0 for i in range(1,width+1)]
    return [pat.count(i) for i in range(1, width+1)]

def s2p(state):
    '''
    Convert state tp pattern number.
    eg [3,4] width 6 <=> [0,0,1,1,0,0]
    '''
    ll = [[i+1]*state[i] for i in range(len(state)) if state[i] != 0]
    return [i for l in ll for i in l]


#%% pickle results

# all simulation results by Douglas
pickle_results = {}
pickle_path = Path('all_kernels_simulations')

for N in Ns_all:
    for width in [5,6,7]:
        for n_photons in [2,3]:
            for conv in ['Left', 'Cent']:
                try:
                    filename = f'allkernels_iterations5state{conv}w{width}d{width}n_photons{n_photons}N{N}_.pickle'
                    f = open(pickle_path.joinpath(filename), 'rb')
                    dataset_pickle = pickle.load(f)
                    f.close()
                    pickle_results[(conv, width, n_photons, N)] = dataset_pickle
                except FileNotFoundError:
                    print(filename, 'not found')
                except EOFError:
                    print(filename)

# experimental results
results_list = { 
    (n,conv): np.load(
    f'exp_data/allkernels_iterations5state{conv}w6d6n_photons2N{n}_results.npy', allow_pickle=True).item() 
    for n in Ns for conv in ['Left', 'Cent'] 
    }

# 40-200 results w6d6
simulation_list = { 
    (n,conv): pickle_results[(conv, 6, 2, n)]
    for n in Ns_all for conv in ['Left', 'Cent'] 
    }

# load different kernel results
# q: quantum, c: classical, g: gaussian, p: polynomial, l: linear, n: neural tangent kernel

q_accs = get_key_list(results_list, 'q_accs')
q_exp_accs = get_key_list(results_list, 'q_exp_accs')
c_accs = get_key_list(results_list, 'c_accs')
c_exp_accs = get_key_list(results_list, 'c_exp_accs')

l_accs = get_key_list(results_list, 'l_accs')
g_accs = get_key_list(results_list, 'g_accs')
p_accs = get_key_list(results_list, 'p_accs')
n_accs = get_key_list(results_list, 'n_accs')
# g_qc = get_key_list(results_list, 'g_qc')

q_accs_simu = get_key_list(simulation_list, 'q_accs', (9,2))
c_accs_simu = get_key_list(simulation_list, 'c_accs', (9,2))
l_accs_simu = get_key_list(simulation_list, 'l_accs', (9,2))
g_accs_simu = get_key_list(simulation_list, 'g_accs', (9,2))
p_accs_simu = get_key_list(simulation_list, 'p_accs', (9,2))
n_accs_simu = get_key_list(simulation_list, 'n_accs', (9,2))
geod_simu = get_key_list(simulation_list, 'geo_diffs', (9,2))

all_accs_simu = np.stack([q_accs_simu, c_accs_simu, 
                          g_accs_simu, p_accs_simu, l_accs_simu, n_accs_simu])

all_accs_exp = np.stack([q_exp_accs, c_exp_accs])

#%% to save the data in results_list and simulation_list
# for key in results_list.keys():
#     N, convention = key
#     np.save(
#     f'allkernels_iterations5state{convention}w6d6n_photons2N{N}_results.npy', results_list[key])

# for key in simulation_list.keys():
#     N, convention = key
#     with open(pickle_path.joinpath(f'allkernels_iterations5state{convention}w6d6n_photons2N{N}_.pickle'), 'wb') as f:
#         print(simulation_list[key].keys())
#         pickle.dump(simulation_list[key], f)