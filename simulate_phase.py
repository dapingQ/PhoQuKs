#%%
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from discopy.quantum.optics import ansatz, to_matrix

from data_utils import p2s, s2p
from multiprocessing import Pool

width = 6
depth = 6
x = [ 1, 1, 0, 0, 0, 0]
patterns = [[i, j] for i in range(1, width+1)
            for j in range(1, width+1) if i < j]

def chip(params): return ansatz(width, depth, params)
def sampler(x0, x1): return (chip(x0) >> chip(x1).dagger())

N_runs = 500
#%%

def run_noise(id, error):
    '''
    run the simulation with random noise
    id: fidelity
    error: noise level
    '''
    # random phase
    xs = np.random.rand(30)
    # random noise in random positions
    noise = np.zeros_like(xs)
    noise[np.random.randint(30, size=30)] += np.random.normal(scale=error, size=30)        
    # add noise to the phase
    xs_noise = xs + noise
    # create the chip with and without noise
    ran_chip = chip(xs)
    ran_chip_noise = chip(xs_noise)
    # sampling
    out = np.array([id*ran_chip.indist_prob( p2s(p, width), x ) 
                        + (1-id)*ran_chip.dist_prob( p2s(p, width), x ) 
                        for p in patterns])
    out_noise = np.array([id*ran_chip_noise.indist_prob( p2s(p, width), x ) 
                        + (1-id)*ran_chip_noise.dist_prob( p2s(p, width), x ) 
                        for p in patterns])
    return out, out_noise, xs, noise

def run_noise_with_params(params):
    id, error = params
    return run_noise(id, error)

data_noise = {}

id = 1
with Pool(25) as p:
    results = list(tqdm(p.imap(run_noise_with_params, 
                               [(id, noise) for noise in [0.01]*N_runs
                                + [0.005]*N_runs
                                + [0.001]*N_runs
                                + [0.0005]*N_runs
                                + [0.0001]*N_runs
                                ]
                                ),
                               total=5*N_runs))

for i, (out, out_noise, xs, noise) in enumerate(results):
    data_noise[f'run_{i}'] = {
        'out': out,
        'out_noise': out_noise,
        'xs': xs,
        'noise': noise
    }

np.save('resolution_vs_fid.npy', data_noise)
#%%

fids = []
data_noise = np.load('resolution_vs_fid.npy', allow_pickle=True).item()
for result in data_noise.values():
    out = result['out']
    out_noise = result['out_noise']
    out /= out.sum()
    out_noise /= out_noise.sum()

    xs = result['xs']
    noise = result['noise']
    fids.append(np.sum(np.sqrt(out*out_noise)))

fids= np.array(fids).reshape(5, N_runs)
#%%

fig, ax = plt.subplots(5,1, figsize=(2, 4), sharex=True)
logbin = np.logspace(-8, -1, 15)
noise_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
# logbin = [0, 0.99, 0.9999, 0.99999]
for i in range(5):
    h, b, _ = ax[i].hist(1-fids[i], bins=logbin, label=f'noise={noise_list[i]}', alpha=0.5)
    ax[i].text(0.7, 0.6, f'{noise_list[i]:.1e}', transform=ax[i].transAxes)
    ax[i].set_ylabel('Counts')
ax[0].set_title('fidelity vs phase noise')
ax[-1].set_xscale('log')
ax[-1].invert_xaxis()
ax[-1].set_xlabel('1-fidelity')
# ax[-1].set_xticks(logbin[::3])
# ax[-1].set_xticklabels(['0.9', '0.999', '0.99999', '0.99999999', '0.99999'][::-1])
fig.subplots_adjust(hspace=0)
fig.savefig('phase_vs_fid.svg')

# %%

# %%

