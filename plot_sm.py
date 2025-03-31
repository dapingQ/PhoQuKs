#%%
from data_utils import *
#%% fig 7, gram matrix plotting

for conv in ['Cent']:
    for N in [40]:
        for nn in [4]:
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
                    'cmap': 'Greys'
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

#%% fig 3b, transition kernel

cc_exp = np.load('hom_fig.npy')
result_trans = np.load('transition_kernel.npy')

cc_err = np.sqrt(cc_exp[:,14])

# select the dataset N=40, iter=1, conv=Left
N = 40
result = results_list[(N, 'Left')]
nn = 1
X = result['Xs'][nn]
y = result['ys'][nn]
y_labels = [1 if z > np.mean(y) else -1 for z in y]
X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(X, y_labels, range(N), test_size=test_size, random_state=RANDOM_STATE)

acc_trans = []
for half_gram in result_trans:
        gram = half_gram[:,:,0] 
        gram = gram + np.diag(np.ones(N)) + np.transpose(gram)    
        gram_train = np.array([[gram[i, j] for j in indexes_train] for i in indexes_train])
        gram_test = np.array([[gram[i, j] for j in indexes_train] for i in indexes_test])
        # Test quantum/classical kernels
        classifier = svm.SVC(kernel='precomputed', verbose=False)  
        try:
            classifier.fit(gram_train, y_train)
            acc_test = accuracy_score(y_test, classifier.predict(gram_test))
            acc_trans.append(acc_test)
        except:
            print(N, nn)        
acc_trans =  np.insert(acc_trans, 0, q_exp_accs[0,0,nn])
acc_trans =  np.insert(acc_trans, -1, c_exp_accs[0,0,nn])

xl = np.arange(2,7,0.02)

fig, ax = plt.subplots(2,1, figsize=(4.5,2), sharex=True, )
ax[0].errorbar(x=xl, 
               y=cc_exp[:,14]/cc_exp.max(), 
               yerr = cc_err/cc_exp.max(), 
               elinewidth=0,
               fmt='.', 
               color='black', 
               markersize=6,
               markeredgewidth=0,
               alpha=.6)

ax[0].set_xlim((6,4.3))
ax[0].set_xticks(np.arange(4.44, 6.24, 0.3))

ax[1].plot(np.arange(4.44, 6.24, 0.3), acc_trans, '--v', ms=4, markeredgewidth=0, color='black', alpha=.8)

ax[0].set_ylabel('normalized CC')

ax[1].set_ylabel('test accuracy')
fig.subplots_adjust(hspace=0)
ax[1].set_xlabel('translation stage position (mm)')
ax[0].grid(True,axis='x')
ax[1].grid(True,axis='x')

# fig.savefig('kernel_trans.svg')

#%% fig 3a, hom dip

cc_exp_norm =  cc_exp[:,14]/cc_exp.max()

from numpy.random import poisson
from scipy.optimize import curve_fit

def tri_func(x, a, b, c, d):
    return a+b*( np.abs(x-c) - 1/2*( np.abs(x-c-d) + np.abs(x-c+d) ) )
popt, pcov = curve_fit(tri_func, xl, cc_exp_norm, p0=[1, 0, 4.5, 1])

a, b, c, d = popt

hd = cc_exp[:,14]
vmin = hd.min()
vmax = hd.max()

vmin_err = np.std(poisson(np.sqrt(vmin), 10))
vmax_err = np.std(poisson(np.sqrt(vmax), 10))

vis = 1 - vmin/vmax
vis_err = np.sqrt( vmin/vmax**2 + vmin**2/vmax**3)

print('vis', vis)
print('vis std', vis_err)

fig, ax = plt.subplots(figsize=(2,2), )
ax.plot(xl, tri_func(xl, *popt), '--', color='red', lw=.6)
ax.errorbar(x=xl,y=cc_exp_norm, 
            yerr=cc_err/cc_exp.max(), elinewidth=1,
            fmt='-.', color='black', alpha=.6)
ax.set_xlim((6.3,2.4))
ax.set_ylabel('normalized CC')
ax.set_xlabel('delay (mm)')

ax.set_yticks(np.arange(0,1.1,0.5))

# ax.plot(xl, a*(c)+b*len(xl), '--', color='green')
# plt.show()
# fig.savefig('hom_fit.svg')


#%% fig 6, fidelity histrogram

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
# ax[1].set_yscale('log')
print(f'quantum kernel mean fid {np.nanmean(qf):.6f}')
print(f'quantum kernel fid std {np.nanstd(qf):.6f}')
print(f'classical kernel mean fid {np.nanmean(cf):.6f}')
print(f'classical kernel fid std {np.nanstd(cf):.6f}')
ax[0].set_title(r'$K_Q$')
ax[1].set_title(r'$K_C$')
fig.subplots_adjust(wspace=0)
# np.savez('raw_data.npz', raw)
# fig.savefig('fid_lin.svg')

#%% fig 1a, all simulated geometric difference 

# simulation results
g_simu_flat = np.array([r['geo_diffs'] for r in pickle_results.values()]).flatten()
q_accs_g = np.array([r['q_accs'] for r in pickle_results.values()]).flatten()
c_accs_g = np.array([r['c_accs'] for r in pickle_results.values()]).flatten()

# sort the accs for fitting
srt = np.argsort(g_simu_flat)

c1 = np.polyfit(x=g_simu_flat, y=q_accs_g, deg=3)
c2 = np.polyfit(x=g_simu_flat, y=c_accs_g, deg=3)

fig, ax = plt.subplots(1,1, figsize=(3,2))

args_l = {
    'ls': '--',
    'alpha': .5,
    # 'ms': ,
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

fig.savefig('geod.svg')

#%% fig 1b, model complexity

# phi, w, phot_n, size, iter
s_Q = np.zeros([2,3,2,9,5])
s_C = np.zeros([2,3,2,9,5])

def model_complexity(y, gram):
    '''
    model complexity
    y: labels
    gram: kernel matrix
    '''
    # N = len(y)
    return y.T @ np.linalg.inv(gram) @ y

for phi, conv in enumerate(['Left', 'Cent']):
    for ww, width in enumerate([5,6,7]):
        for pp, n_photons in enumerate([2,3]):
            for ss, N in enumerate(Ns_all):
                try:
                    result = pickle_results[(conv, width, n_photons, N)]
                    q_qc_accs = []
                    c_qc_accs = []
                    for nn in range(5):
                        kq = result['q_grams'][nn]
                        kc = result['c_grams'][nn]
                        
                        X = np.array(result['Xs'][nn])
                        y = np.array(result['ys'][nn])

                        s_Q[phi,ww,pp,ss,nn] = model_complexity(y, kq)
                        s_C[phi,ww,pp,ss,nn] = model_complexity(y, kc)

                except:
                    print('No this dataset')
                    pass
s_Q_flat = s_Q.flatten()
s_C_flat = s_C.flatten()
fig, ax = plt.subplots(1,1, figsize=(3,2))

ax.plot(g_simu_flat, s_C_flat, '.', label='c', color = '#ff5500', **args_d)
ax.plot(g_simu_flat, s_Q_flat, '.', label='q', color = '#0055ff', **args_d)
ax.set_yscale('log')
# ax.set_ylim(top=10, bottom=0)
ax.set_xlabel(r'geometric difference $g_{CQ}$')
ax.set_ylabel('model complexity')
ax.legend([r'$s_C$', r'$s_Q$'],
          ncol=2, bbox_to_anchor=(.3, 1), loc='lower left', 
          labelspacing=0)
# fig.savefig('model_complexity.svg')
#%%
gd_int = np.linspace(1, 2, 21)
step = 0.04
gd_int = np.arange(1, 2.1, step)

fig, ax = plt.subplots(1,1, figsize=(3,2))
hist_gd = np.zeros(len(gd_int)-1)
for i in range(len(gd_int)-1):
    hist_gd[i] = np.count_nonzero((gd_int[i] < (s_C_flat / s_Q_flat)) & ((s_C_flat / s_Q_flat) < gd_int[i+1]))
ax.bar(gd_int[:-1]+step/2, hist_gd, width=step, alpha=.5, color='green')
ax.set_xlabel(r'$\frac{s_C}{s_Q}$')
ax.set_ylabel('counts')
ax.set_xlim(1, 2)
fig.savefig('model_complexity_hist.svg')

#%% fig 4, sweeping circuit width

fig, ax = plt.subplots(1, 2, figsize=(7,2.8))

data_q = q_cq

error_bar_arg = {
    'capsize': 2,
    'color': 'black'
}

ls_list = ['-o', '-v', '-^']
for c in [0,1]:
    for ww, width in enumerate([5,6,7]): 
        n_photons =2
        pp =1 
        ax[c].errorbar(x=list(Ns_all), 
                        y=np.mean(data_q[c, ww, pp, :, :], axis=1), 
                        yerr=np.std(data_q[c, ww, pp, :, :], axis=1), 
                        **error_bar_arg,
                        fmt=ls_list[ww],
                        label=f'width={width}',
                        alpha=ww*.2+.2 )
        ax[c].set_xlabel('dataset size N')
        ax[c].set_ylabel('accuracy')
        ax[c].legend()

fig.tight_layout()
# fig.savefig('aq_width.svg')

#%% fig 5, Classification accuracies of photonic and Gaussian kernels

fig, ax = plt.subplots(1, 2, figsize=(7,3))

arg_simu = {
    'linestyle': '--', 
    'linewidth':1.2, 
    'alpha':.5,
    'markeredgewidth': 0,
    'markersize': 6,
    'capsize': 2,

}

arg_exp = {
    'linestyle': '-', 
    'linewidth':1.5, 
    'markeredgewidth': 0,
    'markersize': 6,
    'capsize': 2,
    # 'alpha':1
}

for c in [0,1]:
    l_simu =[]
    l_exp = []
    for k in [0,1,2]:
        # simulation dashed
        l_simu.append( ax[c].errorbar(x=Ns_all[:], 
                            y=np.mean(all_accs_simu[k,:,c,:], axis=-1), 
                            yerr=np.std(all_accs_simu[k,:,c,:], axis=-1), 
                            marker=ml[k], c=cl[k],
                            # **arg_simu,
                            **arg_exp,
                            ) )
        # exp solid
        ax[c].set_ylim(top=.9, bottom=0.4)
        # random guessing
        ax[c].plot([40, 200], [.5]*2, '--', lw=1, c='black')
        ax[c].set_xlabel('dataset size N')
        ax[c].set_ylabel('accuracy')

# ax[0].set_ylabel(left_label)
# ax[1].set_ylabel(cent_label)
# ax[1].set_xlabel('dataset size N')

ax[0].legend(l_simu+l_exp,
             ['quantum', 'classical', 'gaussian', 'polynomial', 'linear'] + ['quantum exp.','classical exp.'], 
             ncol=7, frameon=False, bbox_to_anchor=(0.3, 1),
              loc='lower left', columnspacing=0.3
              )
# fig.subplots_adjust(hspace=0)
# fig.savefig('qcg.svg')

#%% fig 2, unbunching

n_iterations = 5
widths = [3, 4, 5, 6, 7]
dps = [40, 60, 80, 100, 120, 140, 160, 180, 200]
separation_kernel = "indist"

# Example datapoint to look at file
ex_width = 4
ex_depth = ex_width
ex_N = 100
ex_separation_kernel = "indist"
ex_filename = "iterations{}w{}d{}N{}sepkernel{}.pickle".format(n_iterations, ex_width, ex_depth, ex_N, ex_separation_kernel)

location  = Path('..//..//photonic-kernels//unbunching_experiments')
path = location.joinpath(ex_filename)
with open(path, 'rb') as f:
    d = pickle.load(f)

# Plot accuracies for the different kernels
fig, axs = plt.subplots(len(widths), 1, figsize=(5, 1.3*len(widths)), sharex=True, facecolor="w")
cmap = plt.get_cmap("tab10")
colors = iter([cmap(i) for i in range(5)])

from cycler import cycler
import os
error_bar_arg = {
    'fmt': '-o',
    'capsize': 2
}

for w, width in enumerate(widths):
    ax = axs[w]
    depth = width
    indist_accs_plot = []
    ub_accs_plot = []
    c_accs_plot = []
    indist_errs = []
    ub_errs = []
    c_errs = []
    eigvs_assertion_fails = []
    geo_diff_assertion_fails = []

    for N_dps in dps:

        args = [n_iterations, width, depth, N_dps, separation_kernel]
        filename = "iterations{}w{}d{}N{}sepkernel{}.pickle".format(n_iterations, width, depth, N_dps, separation_kernel)
        path = location.joinpath( filename.format(args))

        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()

        indist_accs = np.array(data['indist_accs'])
        ub_accs = np.array(data['ub_accs'])
        c_accs = np.array(data['c_accs'])

        indist_mean = np.mean(indist_accs)
        ub_mean = np.mean(ub_accs)
        c_mean = np.mean(c_accs)

        indist_err = np.std(indist_accs, ddof=1)/np.sqrt(n_iterations)
        ub_err = np.std(ub_accs, ddof=1)/np.sqrt(n_iterations)
        c_err = np.std(c_accs, ddof=1)/np.sqrt(n_iterations)

        indist_accs_plot.append(indist_mean)
        indist_errs.append(indist_err)
        ub_accs_plot.append(ub_mean)
        ub_errs.append(ub_err)
        c_accs_plot.append(c_mean)
        c_errs.append(c_err)

        eigvs_assertions = data['eigvs_assertions']
        geo_diff_assertions = data['geo_diff_assertions']
        if not np.all(eigvs_assertions):
            eigvs_assertion_fails.append(N_dps)
        if not np.all(geo_diff_assertions):
            geo_diff_assertion_fails.append(N_dps)
            print(width, N_dps, geo_diff_assertions)

    print(eigvs_assertion_fails)
    print(geo_diff_assertion_fails)

    ax.errorbar(dps, indist_accs_plot, yerr=indist_errs, marker=".", color='#0055ff', label="indistinguishable", **error_bar_arg)
    ax.errorbar(dps, ub_accs_plot, yerr=ub_errs, marker=".", linestyle="--", color='grey', label="unbunching", **error_bar_arg)
    ax.errorbar(dps, c_accs_plot, yerr=c_errs, marker=".", linestyle=":", color='#ff5500', label="distinguishable", **error_bar_arg)
    # ax.set_title(f"width={width}")
    ax.set_ylabel('test accuracy\n'+f'width={width}')
    # ax.set_xlabel('dataset size')
    if width == min(widths):
        ax.legend(loc='lower left',frameon=False, bbox_to_anchor=(0, 1),
                     columnspacing=0.3, ncols=3)

# fig.suptitle(f"{separation_kernel} kernel")
plt.xlim(30, 210)
plt.xlabel("dataset size")
# fig.tight_layout()
fig.subplots_adjust(hspace=0)
# plt.savefig(f"accuracies_{separation_kernel}_full.svg")
# plt.show()


#%% process raw data

raw_data = Path('..//raw_data')

datafiles = [
    # not optimal iter
    '230623_job_0.npy',
    '230623_job_5450.npy',
    '230623_job_10000.npy',
    '230623_job_15000.npy',
    '230623_job_18000.npy',

    '230627_job_22000.npy',
    '230628_job_26000.npy',

    '230629_job_29500.npy',
    '230630_job_32000.npy',
    '230701_job_35500.npy',
    '230701_job_39000.npy',

    '230702_job_rerun_best_0.npy',
    '230702_job_rerun_best_4500.npy',
    '230704_job_rerun_best_8000.npy',
    '230808_dist_left_hc.npy',
    # ]

# left indist
# datafiles = [
    '230704_indist_left_0.npy',
    '230711_indist_left_3000.npy',
    '230712_indist_left_7000.npy',
    '230712_indist_left_7000_2.npy',
    '230712_indist_left_8500.npy',
    '230713_indist_left_12300.npy',
    '230713_indist_left_15000.npy',
    '230715_indist_left_18500.npy',
    '230716_indist_left_23000.npy',
    '230716_indist_left_27000.npy',
    '230716_indist_left_30000.npy', 
    '230717_indist_left_33100.npy',
    '230718_indist_left_37000.npy',
    '230718_indist_left_39855.npy',
    '230718_indist_left_41000.npy',
    '230719_indist_left_45000.npy',
    '230719_indist_left_49000.npy',
    '230720_indist_left_52500.npy',
      '230808_indist_left_hc.npy',
#   ]

# cent distinguish
# datafiles = [
    '230710_dist_cent_0.npy',
    '230720_dist_cent_4000.npy',
    '230721_dist_cent_1560.npy',
    '230721_dist_cent_7800.npy',
    '230721_dist_cent_10000.npy',
    '230722_dist_cent_13500.npy',
    '230722_dist_cent_17000.npy',
    '230723_dist_cent_20500.npy',
    '230723_dist_cent_24500.npy',
    '230724_dist_cent_28000.npy',
    '230724_dist_cent_31000.npy',
    '230725_dist_cent_35000.npy',
    '230726_dist_cent_39000.npy',
    '230726_dist_cent_40500.npy',
    '230727_dist_cent_44500.npy',
    '230727_dist_cent_45500.npy',
    '230728_dist_cent_49500.npy',
    '230728_dist_cent_52970.npy',
    '230808_dist_cent_not_measure.npy',    
    '230808_dist_cent_hc.npy',
    '230808_dist_cent_hc_2.npy',

# cent indist
    '230729_indist_cent_0.npy',
    '230729_indist_cent_3500.npy',
    '230730_indist_cent_7500.npy',
    '230730_indist_cent_11000.npy',

    '230801_indist_cent_17500.npy',
    '230802_indist_cent_21400.npy',
    '230802_indist_cent_25000.npy',
    '230803_indist_cent_26340.npy',
    '230803_indist_cent_30000.npy',

    '230804_indist_cent_33600.npy',
    '230804_indist_cent_37600.npy',
    '230805_indist_cent_41700.npy',
    '230805_indist_cent_45000.npy',
    '230806_indist_cent_49000.npy',
    '230806_indist_cent_53000.npy',
    '230807_indist_cent_not_measured.npy',
    '230807_indist_cent_not_measured_2.npy',

    '230808_indist_cent_hc.npy'
]

full_data = np.array([], dtype=data_dt)
for i in range(len(datafiles)):
    filename = datafiles[i]
    print(filename)
    data_to_load = np.load(raw_data.joinpath(filename))
    full_data = np.append(full_data, data_to_load)
print(full_data.shape)
full_cc = full_data['counts']['cc']
print(full_cc.shape)
full_cc_sum_ch = np.sum(full_cc, axis=2)
full_cc_mean_t = np.mean(full_cc_sum_ch, axis=1)
print(full_cc_mean_t.shape)

plt.plot(full_cc_mean_t,'*')
plt.savefig('full_cc_mean_t.svg')
# plt.ylim(0, 3000)
# %%
