#%% basic functions
from cProfile import label
import time, socket, os, pathlib, sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm
from scipy.stats import unitary_group
from scipy.linalg import inv, eigh
import matplotlib.pyplot as plt
from cycler import cycler

cl = ['#0055ff', '#ff5500', 'grey', 'orange', 'violet']
plt.rcParams.update({    
    # 'figure.max_open_warning': 0,
    # 'figure.subplot.hspace': 0,
    'axes.prop_cycle': cycler('color', cl),
    'axes.labelsize': 8,
    # 'axes.grid': True,
    'axes.titlesize': 8,
    # 'lines.linewidth': 0.8,
    'lines.markersize': 4.0,
    # 'text.usetex': True,
    'font.size':8,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'mathtext.fontset': 'cm',
    'legend.framealpha': 1,
    'legend.frameon': False,
    'legend.edgecolor': 'inherit',
    # 'legend.fancybox': True,
    # 'grid.linestyle': ':',
    'savefig.transparent': True,
    # 'savefig.bbox': 'tight',
    'savefig.dpi': 400,
    'image.cmap': 'Spectral',#'seismic',
    'pdf.fonttype': 42,
})

from discopy.quantum.optics import ansatz, params_shape, to_matrix
from discopy.quantum.optics import npperm


from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import time, pickle

width=6
depth=6

Ns_all = range(40,220,20)
Ns = range(40,120,20)

test_size = 1 / 3
RANDOM_STATE = 42

def chip(params): return ansatz(width, depth, params)
def sampler(x0, x1): return (chip(x0) >> chip(x1).dagger())

c_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).dist_prob(state, state)
q_kernel = lambda state: lambda x0, x1: (chip(x0) >> chip(x1).dagger()).indist_prob(state, state)
g_kernel = lambda gamma: lambda x0, x1: np.exp(- gamma * norm(x0 - x1) ** 2)

def gram_matrix(kernel, x):
    """Build a symmetric positive definite matrix"""
    N = len(x)
    gram = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            gram[i, j] = kernel(x[i], x[j])
    return gram + np.diag(np.ones(N)) + np.transpose(gram)

def gram_matrix_from_result(result, exp_theo, nn, i):
    gram = result[exp_theo][nn,:,:,i]
    return gram + np.diag(np.ones(N)) + np.transpose(gram)

def plot_bar(xdata, xcord=None, xtick=None, *args):
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

def get_key_list(dicts, key, shape_pre = (4,2)):
    nd = np.array([ list(d[key]) for d in dicts.values()])
    # print(nd.shape)
    nd.shape = shape_pre + nd.shape[1:]
    return nd
    
# gram classifier 

def gram_classifier(X, y, gram, N):
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

left_label = 'Left state \n'+r'$\vert 1,1,0,0,0,0 \rangle$'+'\n test accuracy'
cent_label = 'Cent state \n'+r'$\vert 0,0,1,1,0,0 \rangle$'+'\n test accuracy'

#%% pickle results

simulations_pickle = {}
pickle_path = pathlib.Path.cwd().joinpath('simulations')

for N in Ns_all:
    for width in [5,6,7]:
        for n_photons in [2,3]:
            for conv in ['Left', 'Cent']:
                try:
                    filename = f'allkernels_iterations5state{conv}w{width}d{width}n_photons{n_photons}N{N}_.pickle'
                    f = open(pickle_path.joinpath(filename), 'rb')
                    dataset_pickle = pickle.load(f)
                    f.close()
                    simulations_pickle[(conv, width, n_photons, N)] = dataset_pickle
                except FileNotFoundError:
                    print(filename, 'not found')
                except EOFError:
                    print(filename)

# experimental results
results_list = { 
    (n,conv): np.load(
        pathlib.Path.cwd().joinpath('experiments').joinpath(f'allkernels_iterations5state{conv}w6d6n_photons2N{n}_results.npy'), 
        allow_pickle=True).item() 
    for n in Ns for conv in ['Left', 'Cent'] 
    }

# sumulations
# select dataset size 40-200, width 6 depth 6 results 
simulation_list = { 
    (n,conv): simulations_pickle[(conv, 6, 2, n)]
    for n in Ns_all for conv in ['Left', 'Cent'] 
    }

# unpack the results
q_accs = get_key_list(results_list, 'q_accs')
q_exp_accs = get_key_list(results_list, 'q_exp_accs')
c_accs = get_key_list(results_list, 'c_accs')
c_exp_accs = get_key_list(results_list, 'c_exp_accs')
l_accs = get_key_list(results_list, 'l_accs')
g_accs = get_key_list(results_list, 'g_accs')
p_accs = get_key_list(results_list, 'p_accs')
g_qc = get_key_list(results_list, 'g_qc')

q_accs_simu = get_key_list(simulation_list, 'q_accs', (9,2))
c_accs_simu = get_key_list(simulation_list, 'c_accs', (9,2))
l_accs_simu = get_key_list(simulation_list, 'l_accs', (9,2))
g_accs_simu = get_key_list(simulation_list, 'g_accs', (9,2))
p_accs_simu = get_key_list(simulation_list, 'p_accs', (9,2))

#%% all simulated geometric difference 

# all simulation results in the same size array
g_simu_flat = np.array([r['geo_diffs'] for r in simulations_pickle.values()]).flatten()
q_accs_g = np.array([r['q_accs'] for r in simulations_pickle.values()]).flatten()
c_accs_g = np.array([r['c_accs'] for r in simulations_pickle.values()]).flatten()

# sort the index
srt = np.argsort(g_simu_flat)

# polynomial fit
c1 = np.polyfit(x=g_simu_flat, y=q_accs_g, deg=3)
c2 = np.polyfit(x=g_simu_flat, y=c_accs_g, deg=3)

fig, ax = plt.subplots(1,1, figsize=(4,2))

args_l = {
    'ls': '--',
    'alpha': .5,
}
args_d = {
    'ms': 1,
}

ax.plot( g_simu_flat[srt], np.poly1d(c1)(g_simu_flat)[srt], label='q', color = '#0055ff', **args_l)
ax.plot( g_simu_flat[srt], np.poly1d(c2)(g_simu_flat)[srt], label='c', color = '#ff5500', **args_l)

l2, = ax.plot( g_simu_flat, c_accs_g, '.', label='c', color = '#ff5500', **args_d )
l1, = ax.plot( g_simu_flat, q_accs_g, '.', label='q', color = '#0055ff', **args_d )

ax.legend([l1, l2], 
          [r'$a_Q$', r'$a_C$'], 
        #   [r'$K_Q$', r'$K_C$'], 
        #   frameon=True, 
          ncol=2, bbox_to_anchor=(.3, 1), loc='lower left', 
          labelspacing=0)
ax.set_ylabel('accuracy')
ax.set_xlabel(r'geometric difference $g_{CQ}$')

#%% benchmark with NTK

import jax.numpy as jnp

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax
import pathlib

def loss_fn(predict_fn, ys, t, xs=None):
  mean, cov = predict_fn(t=t, get='ntk', x_test=xs, compute_cov=True)
  mean = jnp.reshape(mean, mean.shape[:1] + (-1,))
  var = jnp.diagonal(cov, axis1=1, axis2=2)
  ys = jnp.reshape(ys, (1, -1))

  mean_predictions = 0.5 * jnp.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2,
                                   axis=1)

  return mean_predictions

nkt_accs = []
for key in list(simulation_list.keys())[:]:
    N, conv = key
    data  = simulation_list[key]
    print(N, conv)

    for nn in range(5):
        
        Xs = data['Xs'][nn]
        y = data['ys'][nn]

        y_labels = [1 if z > jnp.mean(y) else -1 for z in y]

        X_train, X_test, y_train, y_test = train_test_split(Xs, y_labels, test_size=1/3, random_state=42)

        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test)

        y_test = jnp.reshape(y_test, (-1,1))
        y_train = jnp.reshape(y_train, (-1,1))

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(30, W_std=1.5, b_std=0.05), 
            stax.Erf(),
            stax.Dense(30, W_std=1.5, b_std=0.05), 
            stax.Erf(),
        )

        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, 
                                                              y_train, diag_reg=1e-4)
        y_g, cov = predict_fn(x_test=X_test, get='ntk', compute_cov=True)
        acc_test = accuracy_score(y_test, jnp.sign(y_g))
        print(acc_test)
        nkt_accs.append(acc_test)


nkt_accs = np.array(nkt_accs).reshape(9,2,5)

#%% mean accs and max iter

max_iter = np.array(
    [[1, 2],
    [2, 2],
    [0, 2],
    [0, 4]])

# nn, N, conv, kk
all_accs_simu = np.stack([q_accs_simu, c_accs_simu, 
                          g_accs_simu, p_accs_simu, 
                          l_accs_simu, nkt_accs])

all_accs_exp = np.stack([q_exp_accs, c_exp_accs])
ml = 'ov^<>*'
cl = ['#0055ff', '#ff5500', 'grey', 'orange', 'violet', 'green']
fig, ax = plt.subplots(2, 1, figsize=(5,5), sharex=True)

arg_simu = {
    'capsize': 2,
    'linestyle': '--', 
    'linewidth':1.2, 
    'alpha':.5,
}

arg_exp = {
    'capsize': 2,
    'linewidth':1.5, 
}
PLOT_EXP = True

for c in [0,1]:
    l_simu =[]
    l_exp = []
    for k in [0,1,2,3,4,5]:
        # simulation dashed
        l_simu.append( ax[c].errorbar(x=Ns_all[:4], 
                            y=np.mean(all_accs_simu[k,:4,c,:], axis=-1), 
                            yerr=np.std(all_accs_simu[k,:4,c,:], axis=-1), 
                            marker=ml[k], c=cl[k],
                            **arg_simu,
                            ) )
        # exp solid
        if k <2 and PLOT_EXP:
            l_exp.append( ax[c].errorbar(Ns, 
                                np.mean(all_accs_exp[k,:,c,:], axis=-1),
                                yerr=np.std(all_accs_exp[k,:,c,:], axis=-1), 
                                marker=ml[k], c=cl[k],
                                **arg_exp) )

ax[0].set_xlabel('dataset size N')
ax[1].set_xlabel('dataset size N')

ax[0].legend(l_simu+l_exp,
              ['quantum', 'coherent', 'gaussian', 
               'polynomial', 'linear', 'ntk'] 
               + ['quantum exp.','coherent exp.'], 
             ncol=4, frameon=False, bbox_to_anchor=(0, 1),
              loc='lower left', columnspacing=0.3
              )

#%%
# single figure 

fig, ax = plt.subplots(1, 1, figsize=(5,2))

# change here to plot another state, 'left' if c==0 else 'cent'
c=0

PLOT_EXP=True

l_simu =[]
l_exp = []

# for k in [0,1]:
for k in range(5):
    # plots simulation results in dashed
    l_simu.append( ax.errorbar(x=Ns, 
                    y=np.mean(all_accs_simu[k,:4,c,:], axis=-1), 
                    yerr=np.std(all_accs_simu[k,:4,c,:], axis=-1), 
                    marker=ml[k], c=cl[k],
                    **arg_simu,
                    ) )
    # plot experimental results in solid
    if k <2 and PLOT_EXP:
        l_exp.append( ax.errorbar(Ns, 
                                y=np.mean(all_accs_exp[k,:,c,:], axis=-1),
                                yerr=np.std(all_accs_exp[k,:,c,:], axis=-1), 
                                marker=ml[k], c=cl[k],
                                    **arg_exp) )

ax.set_xlabel('dataset size N')
ax.set_ylabel('test accuracy')

ax.legend(l_simu+l_exp,
             ['quantum', 'classical', 'gaussian', 'polynomial', 'linear'][:len(l_simu)] + ['quantum exp.','classical exp.'], 
             ncol=7, frameon=False, bbox_to_anchor=(0, 1),
              loc='lower left', columnspacing=0.3
              )
conv = 'left' if c==0 else 'cent'

#%% gram matrix plotting

gram_args = {
        'vmax': 1,
        'vmin': 0,
        'cmap': 'Grays'
    }

for conv in ['Cent', 'Left']:
    for N in [40]: # choose the dataset size
        for nn in range(5): # choose the iteration
            result = results_list[(N, conv)]
            
            y = result['ys'][nn]
            X = result['Xs'][nn]
            y_labels = [1 if z > np.mean(y) else -1 for z in y]

            X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(X, y_labels, range(N), test_size=test_size, random_state=RANDOM_STATE)
            
            # sort the test and train index
            # xx = np.argsort(y_labels) if srt == True else np.arange(len(y))
            kqe, kq, kce, kc = result['q_gram_exp'][nn,:,:], result['q_gram'][nn,:,:], result['c_gram_exp'][nn,:,:], result['c_gram'][nn,:,:]
                        
            gram_args = {
                    'vmax': np.mean([kqe, kq, kce, kc])*3,
                    # 'vmax': 1,
                    'vmin': 0,
                    'cmap': 'Grays'
                }
            fig, ax = plt.subplots(4,2, figsize=(4,5.5), sharex=True, height_ratios=[len(indexes_train), len(indexes_train), len(indexes_test), len(indexes_test)])
            

            ax[0,0].imshow( kqe[indexes_train][:,indexes_train]-np.eye(len(X_train)), **gram_args)
            ax[0,1].imshow( kce[indexes_train][:,indexes_train]-np.eye(len(X_train)), **gram_args)
            ax[1,0].imshow( kq[indexes_train][:,indexes_train]-np.eye(len(X_train)) , **gram_args)
            ax[1,1].imshow( kc[indexes_train][:,indexes_train]-np.eye(len(X_train)) , **gram_args)

            ax[0,0].set_title(r'$K_Q$'+f' exp. N={N}, iter{nn}, {conv}')
            ax[0,1].set_title(r'$K_C$'+f' exp. N={N}, iter{nn}, {conv}')
            ax[1,0].set_title(r'$K_Q$'+f' theo. N={N}, iter{nn}, {conv}')
            ax[1,1].set_title(r'$K_C$'+f' theo.  N={N}, iter{nn}, {conv}')
            
            ax[2,0].imshow( kqe[:,indexes_test][indexes_train].T, **gram_args)
            ax[2,1].imshow( kce[:,indexes_test][indexes_train].T, **gram_args)
            ax[3,0].imshow( kq[:,indexes_test][indexes_train].T , **gram_args)
            psm = ax[3,1].imshow( kc[:,indexes_test][indexes_train].T , **gram_args)

            ax[2,0].set_title(r'$K_Q$'+f' exp. N={N}, iter{nn}, {conv}')
            ax[2,1].set_title(r'$K_C$'+f' exp. N={N}, iter{nn}, {conv}')
            ax[3,0].set_title(r'$K_Q$'+f' theo. N={N}, iter{nn}, {conv}')
            ax[3,1].set_title(r'$K_C$'+f' theo.  N={N}, iter{nn}, {conv}')
            # fig.subplots_adjust(wspace=.05*cm, hspace=.15*cm)
            fig.tight_layout()
            fig.colorbar(psm, ax=ax, orientation='vertical', pad=0.05, aspect=40)
            # fig.savefig(f'all_grams/gram_{N}_{nn}_{conv}.svg')


#%% fidelity analysis
            
raw = {}

# reduce the net numpy to single dim
def compress(d):
    assert d.shape[1] == d.shape[2]
    N = d.shape[1]
    ll = []
    for nn in range(d.shape[0]):
        for i in range(d.shape[1]):
            for j in range(i):
                ll.append(d[nn,i,j])
    return np.array(ll)

for i, key in enumerate(results_list.keys()):
    raw[f'D{i}']={}

    q_cc_exp = results_list[key]['q_cc_exp']
    raw[f'D{i}']['q_cc_exp'] = compress(q_cc_exp)

    q_cc_theo = results_list[key]['q_cc_theo']
    raw[f'D{i}']['q_cc_theo'] = compress(q_cc_theo)

    c_cc_exp = results_list[key]['c_cc_exp']
    raw[f'D{i}']['c_cc_exp'] = compress(c_cc_exp)

    c_cc_theo = results_list[key]['c_cc_theo']
    raw[f'D{i}']['c_cc_theo'] = compress(c_cc_theo)

    raw[f'D{i}']['q_fid'] = compress(np.sum(np.sqrt(q_cc_theo*q_cc_exp), axis=3))
    raw[f'D{i}']['c_fid'] = compress(np.sum(np.sqrt(c_cc_theo*c_cc_exp), axis=3))

fig, ax = plt.subplots(1,2, figsize=(4,2), sharey=True)

qf = raw[f'D{i}']['q_fid']
cf = raw[f'D{i}']['c_fid']

ax[0].hist(qf.ravel(), bins=np.linspace(0.9,0.99,10), alpha=.8)
ax[1].hist(cf.ravel(), bins=np.linspace(0.97,0.999,10), color='#ff5500', alpha=.8)

print(f'quantum kernel mean fid {np.nanmean(qf):.6f}')
print(f'quantum kernel fid std {np.nanstd(qf):.6f}')
print(f'classical kernel mean fid {np.nanmean(cf):.6f}')
print(f'classical kernel fid std {np.nanstd(cf):.6f}')
ax[0].set_title(r'$K_Q$')
ax[1].set_title(r'$K_C$')
fig.subplots_adjust(wspace=0)
# fig.savefig('fid_lin.svg')
