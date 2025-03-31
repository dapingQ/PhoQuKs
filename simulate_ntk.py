#%% neural tangent kernel

import jax.numpy as jnp

from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap


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
#%% neural tangent kernel
n_accs = []
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

        
        # gram_train = kernel_fn(X_train, X_train, 'ntk')
        # gram_test = kernel_fn(X_test, X_train, 'ntk')

        # # Test quantum/classical kernels
        # classifier = svm.SVC(kernel='precomputed', verbose=False)  
        # classifier.fit(gram_train, y_train)

        # acc_test = accuracy_score(y_test, classifier.predict(gram_test))
        # # print(N, conv, acc_test)
        # n_accs.append(acc_test)

        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, 
                                                              y_train, diag_reg=1e-4)
        y_g, cov = predict_fn(x_test=X_test, get='ntk', compute_cov=True)
        acc_test = accuracy_score(y_test, jnp.sign(y_g))
        print(acc_test)
        n_accs.append(acc_test)


n_accs = np.array(n_accs).reshape(9,2,5)


# plt.plot(Ns_all, np.mean(n_accs[:,0,:],axis=1), 'x')
# plt.plot(Ns_all, np.mean(n_accs[:,1,:],axis=1), 'x')
#%%

# ts = jnp.arange(0, 10 ** 3, 10 ** -1)
# ntk_train_loss_mean = loss_fn(predict_fn, y_train, ts)
# ntk_test_loss_mean = loss_fn(predict_fn, y_test, ts, X_test)

# plt.subplot(1, 2, 1)

# plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
# plt.loglog(ts, ntk_test_loss_mean, linewidth=3)

# plt.xlabel('step')
# plt.ylabel('loss')
# plt.legend(['Infinite Train', 'Infinite Test'])

# plt.tight_layout()
# plt.savefig('loss.svg')