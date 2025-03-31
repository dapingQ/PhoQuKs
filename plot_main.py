#%%
from data_utils import *

#%% mean accs and max iter

ml = 'ov^<>*'
cl = ['#0055ff', '#ff5500', 'grey', 'orange', 'violet', 'green']

fig, ax = plt.subplots(2, 1, figsize=(5,5), sharex=True)

arg_simu = {
    'linestyle': '--', 
    'linewidth':1.2, 
    'alpha':.5,
}

arg_exp = {
    'linestyle': '-', 
    'linewidth':1.5, 
}

PLOT_EXP = True

for c in [0,1]: # left, cent
    l_simu =[]
    l_exp = []
    # for k in [0,1]:
    for k in [0,1,2,3,4,5]:
        # simulation dashed
        l_simu.append( ax[c].errorbar(x=Ns_all[:4], 
                            y=np.mean(all_accs_simu[k,:4,c,:], axis=-1), 
                            yerr=np.std(all_accs_simu[k,:4,c,:], axis=-1), 
                            marker=ml[k], c=cl[k],
                            **arg_simu,
                            # **arg_exp,
                            ) )
        # exp solid
        if k <2 and PLOT_EXP:
            l_exp.append( ax[c].errorbar(Ns, 
                                np.mean(all_accs_exp[k,:,c,:], axis=-1),
                                yerr=np.std(all_accs_exp[k,:,c,:], axis=-1), 
                                marker=ml[k], c=cl[k],
                                **arg_exp) )
        # ax[c].set_ylim(top=.9, bottom=0.4)
        # random guessing
        # ax[c].plot([120, 200], [.5]*2, '--', lw=1, c='black')

ax[1].set_xlabel('dataset size N')

ax[0].legend(l_simu+l_exp,
            #  ['quantum', 'classical', 'gaussian', 'polynomial', 'linear'] + ['quantum exp.','classical exp.'], 
              ['quantum', 'coherent', 'gaussian', 'polynomial', 'linear', 'ntk'] + ['quantum exp.','coherent exp.'], 
             ncol=4, frameon=False, bbox_to_anchor=(0, 1),
              loc='lower left', columnspacing=0.3
              )

# fig.savefig('simu_all.svg')

#%% fig 4a-b, select iteration

max_iter = np.array(
    [[1, 2],
    [2, 2],
    [0, 2],
    [0, 4]])

PLOT_EXP=True
for c in [0,1]:
    fig, ax = plt.subplots(1, 1, figsize=(2,2))
    l_simu =[]
    l_exp = []

    for k in [0,1]:
        # simulation dashed
        l_simu.append( ax.errorbar(x=Ns, 
                        y = all_accs_simu[k, range(4), c, max_iter[:,c]],
                        marker=ml[k], c=cl[k],
                        **arg_simu,
                        # **arg_exp,
                        ) )
        # exp solid
        if k <2 and PLOT_EXP:
            l_exp.append( ax.errorbar(Ns, 
                                    y=all_accs_exp[k, range(4), c, max_iter[:,c]],
                                    marker=ml[k], c=cl[k],
                                        **arg_exp) )

    ax.set_xlabel('dataset size N')
    ax.set_ylabel('test accuracy')

    ax.legend(l_simu+l_exp,
                ['quantum', 'classical', 'gaussian', 'polynomial', 'linear'][:len(l_simu)] + ['quantum exp.','classical exp.'], 
                ncol=4, frameon=False, bbox_to_anchor=(0, 1),
                loc='lower left', columnspacing=0.3
                )

    conv = 'left' if c==0 else 'cent'
    # fig.savefig(f'{conv}_best.svg')

#%% fig 3b, horizontal q c
result_bs = results_list[(40, 'Left')]

fig, ax = plt.subplots(1,2, figsize=(7,1.5), sharey=True)
nn, i, j = [0, 18, 10 ]

raw_data_q = np.load('raw_data_q.npy')
raw_data_c = np.load('raw_data_c.npy')

def n_err(n_arr):
    '''
    calculate error of probability
    for P = n_i / sum(n)
    here sum(i) = n_i + sum(n_noti)
    dp/dn_i = (sum(n) - n_i) / sum(n)^2
    dp/dn_noti = 1/sum(n)^2
    dp = dp/dn_i * std(n_i) + dp/dn_noti * sum(std(n)) - dp/dn_noti * std(n_i)
    '''
    n_std = np.std(n_arr, axis=0) 
    n_mean = np.mean(n_arr, axis=0) 
    n_err_list = []
    for i in range(15):
        dp_dn_i = (np.sum(n_mean)- n_mean[i])/np.sum(n_mean)**2
        dp_dn_noti = 1/np.sum(n_mean)**2
        dp = dp_dn_i * n_std[i] + dp_dn_noti*np.sum(n_std) - dp_dn_noti*n_std[i]
        n_err_list.append(dp)
    return n_err_list


# def n_err(n_arr):
#     '''
#     calculate error of probability
#     for P = n_i / sum(n)
#     here sum(i) = n_i + sum(n_noti)
#     dp/dn_i = (sum(n) - n_i) / sum(n)^2
#     dp/dn_noti = 1/sum(n)^2
#     dp = dp/dn_i * std(n_i) + dp/dn_noti * std(sum(n)) 
#     '''
#     n_std = np.std(n_arr, axis=0)
#     n_mean = np.mean(n_arr, axis=0)
#     n_sum_std = np.std(np.sum(n_arr, axis=0))
#     n_err_list = []
#     for i in range(15):
#         dp_dn_i = (np.sum(n_mean)- n_mean[i])/np.sum(n_mean)**2
#         dp_dn_noti = 1/np.sum(n_mean)**2
#         dp = dp_dn_i * n_std[i] + dp_dn_noti*n_sum_std
#         n_err_list.append(dp)
#     return n_err_list

c_std = n_err(raw_data_c['counts']['cc'])
print(c_std)
q_std = n_err(raw_data_q['counts']['cc'])
print(q_std)
width = 6
patterns = [[i, j] for i in range(1, width+1)
            for j in range(1, width+1) if i < j]
pattern_str = [str(i)[1:-1].replace(' ', '') for i in patterns]
xcords=np.linspace(0,15,15)

# '#0055ff', '#ff5500'
l1 = ax[0].bar(x=xcords, width = 0.6, height=result_bs['q_cc_exp'][nn,i,j,:], yerr=c_std, capsize=3, color='#0055ff')
l2 = ax[0].bar(x=xcords, width = 0.6, height=result_bs['q_cc_theo'][nn,i,j,:], color='None', edgecolor='black')
l3 = ax[1].bar(x=xcords, width = 0.6, height=result_bs['c_cc_exp'][nn,i,j,:], yerr=q_std, capsize=3, color='#ff5500')
l4 = ax[1].bar(x=xcords, width = 0.6, height=result_bs['c_cc_theo'][nn,i,j,:], color='None', edgecolor='black')

ax[0].set_xticks(xcords)
ax[1].set_xticks(xcords)
ax[0].set_xticklabels(pattern_str, rotation=40)
ax[1].set_xticklabels(pattern_str, rotation=40)

ax[0].legend([l1,l2,l3,l4], 
            #  ['quantum \nexp.', 'quantum\n theo.','classical\n exp.', 'classical\ntheo.'], 
             ['quantum exp.', 'quantum theo.','classical exp.', 'classical theo.'], 
             ncol=4, frameon=False, bbox_to_anchor=(0.1, 1),
             loc='lower left', labelspacing=.2)

ax[0].set_ylabel('Probability')

fig.subplots_adjust(wspace=0)
# fig.savefig('bs_horizontal_err.svg', bbox_inches='tight')

# %%
