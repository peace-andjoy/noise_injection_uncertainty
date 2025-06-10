# %%
# 20250104修改：把模拟数据改成从正弦函数中生成
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import autograd
import autograd.misc.flatten as flatten
import autograd.numpy as ag_np

from bnn_prior import make_nn_params_as_dicts, make_nn_params_as_lists
from bnn_prior import flatten_dicts, flatten_lists, unflatten_vector
from activations import relu, tanh, rbf

np.random.seed(123)
# 定义方程
def equation(x):
    epsilon = np.random.normal(0, np.abs(x))
    return 0.3 * np.sin(x*np.pi) + 0.2 * epsilon

num_samples = 200
# 生成 x 值
x_values = np.linspace(-2, 2, num_samples)

# 计算对应的 y 值
y_values = equation(x_values)

x_train, y_train = x_values, y_values

sigma = 5


def U(q):
    """
    Returns the potential energy (float) of a given
    neural network configuration. Assumes x_train has been set already.

    Arguments:
    - q: A Numpy array corresponding to weights and biases of a NN
    """
    y_preds = ag_np.array([nn_predict(x=x, nn_param_list=q) for x in x_train])
    neg_log_L = ag_np.sum(ag_np.square(y_preds - y_train))/(2*sigma**2)
    neg_log_prior = ag_np.sum(ag_np.square(q))/2
    return neg_log_L + neg_log_prior


grad_U = autograd.grad(U) # Calculate the gradient of potential energy


def K(p):
    """
    Calculates the kinetic energy of some given momentum values and returns
    a float.

    Arguments:
    - p: Numpy array of momentum values
    """
    return ag_np.sum(ag_np.square(p)) / 2

def ag_np_relu(x):
    return ag_np.maximum(0, x)

def nn_predict(x, nn_param_list=None, act=ag_np_relu):
    """
    This neural net prediction is terrible, and only works for the
    architecture with one hidden layer of 10 units. This was partly
    for desperation/debugging purposes, but had a neat side effect
    of making computation blazingly fast.

    Arguments: 
    - x: A scalar to predict on
    - nn_param_list: Numpy array of 31 NN parameters
    - act: Activation function to use (default=tanh)
    """
    h = act(ag_np.dot(x, nn_param_list[:100]) + nn_param_list[100:200])
    y = ag_np.dot(h,nn_param_list[200:300] + nn_param_list[300])
    return y


def run_HMC_sampler(init_bnn_params=None, n_hmc_iters=1000, n_leapfrog_steps=25,
                    eps=0.001, random_seed=42, U=U,
                    K=K, grad_U=grad_U):
    """ Run HMC sampler for many iterations (many proposals)

    Returns
    -------
    bnn_samples : list
        List of samples of NN parameters produced by HMC
        Can be viewed as 'approximate' posterior samples if chain runs to convergence.
    info : dict
        Tracks energy values at each iteration and other diagnostics.

    
    """
    # Create random-number-generator with specific seed for reproducibility
    prng = np.random.RandomState(int(random_seed))

    # Set initial bnn params
    cur_q = init_bnn_params
    cur_U = U(cur_q)

    bnn_samples = []
    energies = []
    energies.append(cur_U)
    start_time_sec = time.time()

    n_accept = 0
    for t in range(n_hmc_iters):

        cur_p = prng.normal(size=301)

        # Create PROPOSED configuration
        prop_q, prop_p = make_proposal_via_leapfrog_steps(
            cur_q, cur_p,
            n_leapfrog_steps=n_leapfrog_steps,
            eps=eps,
            grad_U=grad_U)

        prop_U = U(prop_q)
        cur_K = K(cur_p)
        prop_K = K(prop_p)
        accept_prob = ag_np.exp(cur_U-prop_U+cur_K-prop_K)
      

        # Draw random value from (0,1) to determine if we accept or not
        if prng.rand() < accept_prob:
            n_accept += 1
            cur_q, cur_U = (prop_q, prop_U)

        # Update list of samples from "posterior"

        bnn_samples.append(cur_q)
        energies.append(cur_U)

        # Print some diagnostics every 50 iters
        if t < 5 or ((t+1) % 50 == 0) or (t+1) == n_hmc_iters:
            accept_rate = float(n_accept) / float(t+1)
            print("iter %6d/%d after %7.1f sec | accept_rate %.6f" % (
                t+1, n_hmc_iters, time.time() - start_time_sec, accept_rate))
            
            x_grid = np.linspace(-6,6,100)
            pred = ag_np.array([nn_predict(x=i, nn_param_list=cur_q) for i in x_grid])
            rmse = np.sqrt(np.mean((x_grid**3-pred)**2))
            print("rmse=", rmse)

    return (
        bnn_samples,
        energies,
        dict(
            n_accept=n_accept,
            n_hmc_iters=n_hmc_iters,
            accept_rate=accept_rate),
        )


def make_proposal_via_leapfrog_steps(
        cur_bnn_params, cur_momentum_vec,
        n_leapfrog_steps=25,
        eps=0.001,
        grad_U=grad_U):
    """ Construct one HMC proposal via leapfrog integration

    Returns
    -------
    prop_bnn_params : same type/size as cur_bnn_params
    prop_momentum_vec : same type/size as cur_momentum_vec

    """
    # Initialize proposed variables as copies of current values
    q = copy.deepcopy(cur_bnn_params)
    p = copy.deepcopy(cur_momentum_vec)

    p = p - (eps * grad_U(q) / 2)

    for step_id in range(n_leapfrog_steps):
        q = q + (eps*p)
        if step_id < (n_leapfrog_steps - 1):
            p = p - (eps * grad_U(q))
        else:
            p = -1 * (p - (eps * grad_U(q) / 2))

    return q, p    

def plot_lines(x, lines, train=False):
    """
    Plots one figure with some x, and a bunch of y's from lines
    """
    plt.figure()
    if train:
        plt.plot(x_train, y_train, 'rx')
    for l in lines:
        plt.plot(x, l, '.-')


def multi_predict(x, bnn_configs):
    """
    Returns predictions on x from each of the BNN configrations passed in.
    The returned array is of size (N x S).

    Arguments:
    - x: A Numpy array of inputs of size S
    - bnn_configs: A Numpy array (N x 31)
    """
    multi_preds = []
    for bnn_config in bnn_configs:
        preds = ag_np.array([nn_predict(x=i, nn_param_list=bnn_config) for i in x])
        multi_preds.append(preds)
    return ag_np.array(multi_preds)

def mean_standardized_log_loss(
    y_true, y_pred, y_std, sample_weight=None, multioutput="uniform_average", squared=True
):
    """Mean standardized log loss.
    Read more in the :ref:`User Guide <mean_standardized_log_loss>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated standard deviation in predictions.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    Examples
    --------
    >>> from sklearn.metrics import mean_standardized_log_loss
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> y_std = [0.1, 0, 0.05, 0.3]
    >>> mean_standardized_log_loss(y_true, y_pred, y_std)
    6.356
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> y_std = [[0.01, 0.02],[0.01,0.04],[0.03,0.04]]
    >>> mean_standardized_log_loss(y_true, y_pred, y_std)
    5.511
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([5.00107605, 6.02159874])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    2.858
    """
    # y_type, y_true, y_pred, multioutput = _check_reg_targets(
    #   y_true, y_pred, multioutput
    # )
    # check_consistent_length(y_true, y_pred, sample_weight)
    
    ###########
    # Checks like the above ones to be implemented.
    ###########
    
    first_term = 0.5 * np.log(2 * np.pi * y_std**2)
    second_term = ((y_true - y_pred)**2)/(2 * y_std**2)
    
    output_errors = np.average(first_term + second_term, axis=0, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


if __name__ == '__main__':
    """
    I apologize for not making this into a function. I again can only
    say that I was getting desperate.
    """
    x_grid = np.linspace(-2, 2, num_samples)
    y1 = 0.3 * np.sin(x_grid*np.pi)
    n_std_dev = 1.96
    chains = 1
    chain_length = 1000
    burnin = 50
    samples = 50
    posteriors = []
    energies = []
    for chain in range(chains):
        nn_params = np.random.normal(size=301)
        t1 = time.time()
        post, es, info = run_HMC_sampler(n_hmc_iters=chain_length,
                                    init_bnn_params=nn_params, eps=0.01, n_leapfrog_steps=500)
        t2 = time.time()
        print('time:', t2-t1)
        print(info)
        energies.append(es)
        posteriors.append(post)
        ind = np.random.choice(range(burnin, chain_length), samples, replace=False)
        post_samples = np.array(post)[ind]
        plot_lines(x_grid, (multi_predict(x_grid, post_samples)), train=True)
        plt.savefig('bnn_posterior_' + str(plt.gcf().number) + '.png',
                    bbox_inches='tight')
    plot_lines(range(len(energies[0])), energies)
    plt.savefig('potential_energies.png', bbox_inches='tight')
    func_samples = multi_predict(x_grid, posteriors[0][-600:-100])
    mean_sample = np.mean(func_samples, axis=0)
    std = np.std(func_samples, axis=0)
    plt.figure(figsize=[3, 3], dpi=300)
    plt.plot(x_train, y_train, 'rx')
    plt.plot(x_grid, y1, linestyle='--')
    plt.plot(x_grid, mean_sample, 'k-')
    plt.gca().fill_between(x_grid.flat, mean_sample-n_std_dev*std, mean_sample+n_std_dev*std,
                           color="#dddddd")
    plt.savefig('posterior_samples.png', bbox_inches='tight')
    plt.show()
    
    import tensorflow as tf
    MSE = np.mean((y_values-pred)**2)
    MSLL = mean_standardized_log_loss(y_values, pred, std, sample_weight=None, multioutput="uniform_average", squared=True)

    # hyperparameters
    lambda_ = 0.01 # lambda in loss fn
    alpha_ = 0.05  # capturing (1-alpha)% of samples
    soften_ = 160.
    n_ = batch_size # batch size

    # define loss fn
    def qd_objective(y_true, y_pred):
        '''Loss_QD-soft, from algorithm 1'''
        y_u = y_pred[:,1]
        y_l = y_pred[:,0]
        
        K_HU = tf.maximum(0.,tf.sign(y_u - y_true))
        K_HL = tf.maximum(0.,tf.sign(y_true - y_l))
        K_H = tf.multiply(K_HU, K_HL)
        
        K_SU = tf.sigmoid(soften_ * (y_u - y_true))
        K_SL = tf.sigmoid(soften_ * (y_true - y_l))
        K_S = tf.multiply(K_SU, K_SL)
        
        MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l),K_H))/tf.reduce_sum(K_H)
        PICP_H = tf.reduce_mean(K_H)
        PICP_S = tf.reduce_mean(K_S)
        
        Loss_H = MPIW_c + lambda_ * n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_H)
        Loss_S = MPIW_c + lambda_ * n_ / (alpha_*(1-alpha_)) * tf.maximum(0.,(1-alpha_) - PICP_S)
        
        return Loss_H, Loss_S

    pred_interval = np.stack((pred-n_std_dev*std, pred+n_std_dev*std), axis=1)
    loss_qd_H, loss_qd_S = qd_objective(y_values, pred_interval)

    print("MSE={:.4g}, MSLL={:.4g}, loss_qd={:.4g}, {:.4g}".format(MSE, MSLL, loss_qd_H, loss_qd_S))