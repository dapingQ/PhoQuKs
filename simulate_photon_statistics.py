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
N_samples = [100, 500, 1000, 5000, 10000]

#%%

def run_photon_poisson(id, N_samples):
    '''
    run the simulation with random noise
    id: fidelity
    error: noise level
    '''
    # random phase
    xs = np.random.rand(30)

    # create the chip with and without noise
    ran_chip = chip(xs)
    # sampling results
    out = np.array([id*ran_chip.indist_prob( p2s(p, width), x ) 
                        + (1-id)*ran_chip.dist_prob( p2s(p, width), x ) 
                        for p in patterns])
    N_raw = np.round(N_samples*out)
    N_poisson = np.random.poisson(N_raw)
    
    return N_raw, N_poisson

run_photon_poisson(1, 5000)

#%%


def run_noise_with_params(params):
    id, N_sam = params
    return run_photon_poisson(id, N_sam)

data_noise = {}
N_runs = 500

id = 1
run_params = [(id, N) for N in N_samples for _ in range(N_runs)]

with Pool(25) as p:
    results = list(tqdm(p.imap(run_noise_with_params, 
                               run_params,
                                ),
                               total=len(N_samples)*N_runs))

for i, (N_raw, N_poisson) in enumerate(results):
    data_noise[f'run_{i}'] = {
        'N_raw': N_raw,
        'N_poisson': N_poisson,
    }

np.save('poisson_vs_fid.npy', data_noise)
#%%

fids = []
data_noise = np.load('poisson_vs_fid.npy', allow_pickle=True).item()
for result in data_noise.values():
    N_raw = result['N_raw']
    N_poisson = result['N_poisson']
    out_raw = N_raw/N_raw.sum()
    out_poisson = N_poisson/N_poisson.sum()

    fids.append(np.sum(np.sqrt(out_raw*out_poisson)))

fids= np.array(fids).reshape(len(N_samples), N_runs)

np.mean(fids,axis=1)
#%%

fig, ax = plt.subplots(5,1, figsize=(2, 4), sharex=True)
logbin = np.logspace(-5, -1, 11)
for i in range(5):
    h, b, _ = ax[i].hist(1-fids[i], bins=logbin, label=f'{N_samples[i]}', alpha=0.5, color='green')
    # ax[i].legend()
    ax[i].text(0.7, 0.6, f'{N_samples[i]:.1e}', transform=ax[i].transAxes)
    ax[i].set_ylabel('Counts')

ax[-1].set_xscale('log')
ax[-1].invert_xaxis()
ax[-1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
ax[-1].set_xlabel('1-fidelity')
ax[0].set_title('fidelity vs mean photon number')
fig.subplots_adjust(hspace=0)
fig.savefig('poisson_vs_fid.svg')

# %%

# %%


