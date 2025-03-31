#%%
import numpy as np
from discopy.quantum.optics import Id, Box, params_shape
from discopy.monoidal import PRO
from discopy.matrix import Matrix

class BMZI(Box):
    """
    Biased Mach-Zender interferometer.
    """
    def __init__(self, theta, phi, bias, _dagger=False):
        self.bias = bias
        self.theta, self.phi, self._dagger = theta, phi, _dagger
        super().__init__('BMZI', PRO(2), PRO(2), _dagger=_dagger)

    @property
    def global_phase(self):
        if not self._dagger:
            return 1j * np.exp(1j * self.theta * np.pi)
        else:
            return - 1j * np.exp(- 1j * self.theta * np.pi)

    @property
    def matrix(self):
        '''
        Consider the bias alpha, beta
        which is equivalent to R(beta)*U(theta, phi)*R(alpha)
        here
        R(alpha) = [[cos(alpha), i*e^(-i*phi)*sin(alpha)],
                    [i*e^(i*phi)*sin(alpha), cos(alpha)]]
        R(beta) = [[cos(beta), i*sin(beta)],
                    [i*sin(beta), cos(beta)]]
        '''
        backend = sympy if hasattr(self.theta, 'free_symbols') else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        
        array = np.array([exp * sin, cos, exp * cos, -sin])

        alpha, beta = self.bias
        cos_alpha = backend.cos(backend.pi * alpha)
        sin_alpha = backend.sin(backend.pi * alpha)
        exp_alpha = backend.exp(1j * 2 * backend.pi * self.phi)
        exp_minus_alpha = backend.exp(-1j * 2 * backend.pi * self.phi)

        cos_beta = backend.cos(backend.pi * beta)
        sin_beta = backend.sin(backend.pi * beta)

        R_alpha = np.array([[cos_alpha, 1j * exp_minus_alpha * sin_alpha],
                            [1j * exp_alpha * sin_alpha, cos_alpha]])
        R_beta = np.array([[cos_beta, 1j * sin_beta],
                            [1j * sin_beta, cos_beta]])

        array = np.dot(R_beta, np.dot(array.reshape(2, 2), R_alpha))
        array = array.flatten()

        matrix = Matrix(self.dom, self.cod, array)
        matrix = matrix.dagger() if self._dagger else matrix
        return matrix

    def dagger(self):
        return BMZI(self.bias, self.theta, self.phi, _dagger=not self._dagger)


def ansatz_bias(width, depth, x, bias=None):
    """
    Returns a universal interferometer given width, depth and parameters x,
    based on https://arxiv.org/abs/1603.08788.
    """
    params = x.reshape(params_shape(width, depth))
    bias = bias.reshape(params_shape(width, depth)) if bias is not None else None
    if bias is not None:
        bias_params = bias.reshape(params_shape(width, depth))
    else:
        bias_params = np.zeros_like(params)

    chip = Id(width)
    if not width % 2:
        if depth % 2:
            params, last_layer = params[:-width // 2].reshape(
                params_shape(width, depth - 1)), params[-width // 2:]
            bias_params, last_layer_bias = bias_params[:-width // 2].reshape(
                params_shape(width, depth - 1)), bias_params[-width // 2:]
        for i in range(depth // 2):
            chip = chip\
                >> Id().tensor(*[
                    BMZI(theta=params[i, j][0], 
                         phi=params[i, j][1],
                         bias=bias_params[i, j])
                    for j in range(width // 2)])\
                >> Id(1) @ Id().tensor(*[
                    BMZI(theta=params[i, j + width // 2][0],
                         phi=params[i, j + width // 2][1], 
                         bias=bias_params[i, j + width // 2])
                    for j in range(width // 2 - 1)]) @ Id(1)
        if depth % 2:
            chip = chip >> Id().tensor(*[
                BMZI(theta=last_layer[j][0],
                     phi=last_layer[j][1], 
                     bias=last_layer_bias[j]) for j in range(width // 2)])
    else:
        for i in range(depth):
            left, right = (Id(1), Id()) if i % 2 else (Id(), Id(1))
            chip >>= left.tensor(*[
                BMZI(theta=params[i, j][0], 
                     phi=params[i, j][1],
                     bias=bias_params[i, j])
                for j in range(width // 2)]) @ right
    return chip

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

def chip_bias(params, bias): return ansatz_bias(width, depth, params, bias)
def sampler(x0, x1): return (chip_bias(x0) >> chip_bias(x1).dagger())

N_runs = 500
noise_list = [0.05, 0.03, 0.02, 0.01, 0.005]

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

    # create the chip with and without noise
    ran_chip = chip_bias(xs, bias=np.zeros_like(xs))
    ran_chip_noise = chip_bias(xs, bias=noise)
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
run_params = [(id, noise) for noise in noise_list for _ in range(N_runs)]

with Pool(25) as p:
    results = list(tqdm(p.imap(run_noise_with_params, 
                               run_params,
                                ),
                               total=5*N_runs))

for i, (out, out_noise, xs, noise) in enumerate(results):
    data_noise[f'run_{i}'] = {
        'out': out,
        'out_noise': out_noise,
        'xs': xs,
        'noise': noise
    }

np.save('bs_bias_vs_fid.npy', data_noise)
#%%

fids = []
data_noise = np.load('bs_bias_vs_fid.npy', allow_pickle=True).item()
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
logbin = np.logspace(-6, -1, 13)
for i in range(5):
    h, b, _ = ax[i].hist(1-fids[i], bins=logbin, label=f'{noise_list[i]}', color='orange', alpha=0.5)
    # ax[i].legend()
    ax[i].text(0.7, 0.6, f'{noise_list[i]:.1e}', transform=ax[i].transAxes)
    ax[i].set_ylabel('Counts')

ax[-1].set_xscale('log')
ax[-1].invert_xaxis()
ax[-1].set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
ax[-1].set_xlabel('1-fidelity')
ax[0].set_title('fidelity vs beam splitter bias')
fig.subplots_adjust(hspace=0)
fig.savefig('bs_bias_vs_fid.svg')



# %%
