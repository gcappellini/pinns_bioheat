import os
import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import pandas as pd
from openpyxl import load_workbook
import time
import random
import imgkit
import torch
import wandb
from scipy import integrate
from scipy.interpolate import interp1d

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gp_seed = None
dde_seed = 101
ITERATION = 0

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

epochs = 20000

# General parameters
L0, TM, Ta, tauf, qmet = 0.05, 45, 37, 1800, 4200

# Tissue parameters
# rho, c, k_eff, W_min, W_avg, W_max, cb = 888, 2387, 1.2, 0.36, 0.54, 0.72, 3825           # fat
rho, c, k_eff, W_min, W_avg, W_max, cb = 1050, 3639, 5, 0.45, 2.48, 4, 3825           # muscle

dT = TM - Ta
alfa = rho * c / k_eff

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)


# Antenna parameters
beta, cc, X0, p = 1, 16, 0.09, 76142.131


# Observer parameters
k = 1


# HPO setting
n_calls = 1000
dim_learning_rate = Real(low=0.000012774471609203795, high=0.21788060459464648, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=8, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=7, high=384, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"],
                             name="activation")
# ["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"]
dim_initialization = Categorical(categories=["Glorot uniform", "He normal", "He uniform"],
                             name="initialization")
dim_w_domain = Integer(low=1, high=178, name="w_domain")
dim_w_bcl = Integer(low=1, high=250, name="w_bcl")
dim_w_bcr = Integer(low=1, high=198, name="w_bcr")
dim_w_ic = Integer(low=1, high=182, name="w_ic")


dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization,
    dim_w_ic,
    dim_w_bcl,
    dim_w_bcr,
    dim_w_domain
]

def inizia():
    # Crea la struttura delle cartelle
    cartella_figure = f"{output_path}figures"
    cartella_history = f"{output_path}history"
    cartella_model = f"{output_path}model"

    # Crea le cartelle se non esistono già
    os.makedirs(cartella_figure, exist_ok=True)
    os.makedirs(cartella_history, exist_ok=True)
    os.makedirs(cartella_model, exist_ok=True)

    return output_path


def inizia_hpo():
    global output_path, gp_seed, dde_seed

    if gp_seed is None:
        gp_seed = random.randint(0, 1000)  # Genera il seed per GP
    if dde_seed is None:
        dde_seed = random.randint(0, 1000)  # Genera il seed per DDE

    output_path = f"{folder_path}hpo/{dde_seed}_{gp_seed}/"

    # Crea la struttura delle cartelle
    cartella_figure = f"{output_path}figures"
    cartella_history = f"{output_path}history"
    cartella_model = f"{output_path}model"

    # Crea le cartelle se non esistono già
    os.makedirs(cartella_figure, exist_ok=True)
    os.makedirs(cartella_history, exist_ok=True)
    os.makedirs(cartella_model, exist_ok=True)

    return output_path, gp_seed, dde_seed

# Considero il caso flusso costante
def gen_testdata():
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_pde.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y


def source(s):
    return qmet + beta*torch.exp(-cc*L0*(X0-s))*p


def pde_m(x, theta, W):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=4)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def ic_obs(x):
    y1 = x[0, 1:2]
    y2 = x[0, 2:3]
    y3 = x[0, 3:4]
    b1 =  6# arbitrary parameter
    # a2 = 0
    # e = y1 + (y3 - 2*a2 + k*(y2-y1-a2))*x + a2*x**2
    e = (y3 + k * (y2 - y1))/(b1 * np.cos(b1) + k * np.sin(b1))* np.sin(b1*x) + y1
    return e[:, 0:1]


def bc0_obs(x, theta, X):
    return x[:, 1:2] - theta


def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    return dtheta_x - x[:, 3:4] - k * (x[:, 2:3] - theta) 


def create_observer(config, W):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    xmin = [0, 0, 0, -2]
    xmax = [1, 1, 1, 0]
    geom = dde.geometry.Hypercube(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_obs, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_m(x, theta, W), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [5] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    model = dde.Model(data, net)
    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def train_model(model, name):
    global dde_seed, output_path
    dde.config.set_random_seed(dde_seed)
    # early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)

    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{output_path}model/{name}.ckpt")#,
                                        #    callbacks=[early_stopping])
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{name}_loss",
                 train_fname=f"{name}_train", test_fname=f"{name}_test",
                 output_dir=f"{output_path}history")
    return model



def gen_obsdata():
    data = np.loadtxt(f"{folder_path}output_matlab_pde.txt")
    x = data[:, 0:1]
    x_end = np.min(x)
    theta_end = data[data[:, 0] == x_end]

    unique_x = np.unique(x)
    sorted_unique_x = np.sort(unique_x)[::-1]
    x_sup = sorted_unique_x[1]
    theta_sup = data[data[:, 0] == x_sup]

    x_max = np.max(x)
    theta_extra = data[data[:, 0] == x_max]
    fl = (theta_extra[:, 2:] - theta_sup[:, 2:])/(x_max - x_sup)

    mask = data[:, 0] != x_max
    data = data[mask, :]
    x, t, exact = data[:, 0:1], data[:, 1:2], data[:, 2:]
    y = exact.flatten()[:, None]
    num_rows = data.shape[0]
    new_columns = np.zeros((num_rows, 3))
    X = np.hstack((x, new_columns, t))

    unique_t = np.unique(t)
    for el in range(len(unique_t)):
        a = X[X[:, 4] == unique_t[el]]
        a[:, 1] = theta_end[el, 2:]
        a[:, 2] = theta_sup[el, 2:]
        a[:, 3] = fl[el]
        X[X[:, 4] == unique_t[el]] = a

    return X, y


def train2_model(model, name):
    global dde_seed, output_path
    dde.config.set_random_seed(dde_seed)

    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{output_path}model/{name}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{name}_loss",
                 train_fname=f"{name}_train", test_fname=f"{name}_test",
                 output_dir=f"{output_path}history")
    
    # X, y_true = gen_testdata()
    XO, y_true = gen_obsdata()
    y_pred = model.predict(XO)
    er = dde.metrics.l2_relative_error(y_true, y_pred)

    # pp_theta = np.load(f"{folder_path}pinns/theta.npz")
    # theta = pp_theta['data']
    # er = dde.metrics.l2_relative_error(theta, y_pred)
    
    # train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    # test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    # metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()
    # er = test.min()
    
    return er


def restore_model(model, name):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    model.restore(f"{folder_path}model/{name}.ckpt-{epochs}.pt", verbose=0)
    return model


def configure_subplot(ax, XS, surface):
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))
    X = XS[:, 0:1].reshape(le, la)
    T = XS[:, 1:].reshape(le, la)

    ax.plot_surface(X, T, surface, cmap='inferno', alpha=.8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)


def plot_1obs(o):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    XO, y_true = gen_obsdata()

    XS = np.delete(XO, [1, 2, 3], axis=1)
    sys = y_true
    obs = o.predict(XO)


    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['System', 'Observer', 'Error']

    ll = int(np.sqrt(len(y_true)))
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))


    # Define surfaces for each subplot
    surfaces = [
        [sys.reshape(le, la), obs.reshape(le, la),
         np.abs(sys - obs).reshape(le, la)]
    ]

    # Create a grid of subplots
    grid = plt.GridSpec(1, 3)

    # Iterate over columns to add subplots
    for col in range(3):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        configure_subplot(ax, XS, surfaces[0][col])

        # Set column titles
        ax.set_title(col_titles[col], fontsize=8, y=.96, weight='semibold')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15)

    # Save and show plot
    plt.savefig(f"{folder_path}figures/plot_1obs.png", dpi=300, bbox_inches='tight')
    plt.show()



def plot_1obs_tf(oo):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    XO, y_true = gen_obsdata()

    XOf = XO[XO[:, 4]==1]
    sf = y_true[-len(XOf):, :]
    of = oo.predict(XOf)    

    x = XOf[:, 0:1]

    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', color='C0', label="System", markevery=2)
    ax2.plot(x, of, linestyle='None', marker="x", color='C1', label="Observer", markevery=3)

    ax2.legend()
    ax2.set_ylabel(ylabel=r"Temperature")
    ax2.set_xlabel(xlabel=r"Distance $x$")
    ax2.set_title(r"Solution at $t=t_f$", weight='semibold')

    plt.grid()

    plt.savefig(f"{folder_path}figures/tf_1observer.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_1obs_l2(oo):
    l2_k0 = []
    XO, y_true = gen_obsdata()
    tot = np.hstack((XO, y_true))
    tt = np.unique(tot[:, 4:5])
    for t in tt:
        XOt = tot[tot[:, 4]==t]
        s = XOt[:, 5:]
        o = oo.predict(XOt[:, :5])
        l2_k0.append(dde.metrics.l2_relative_error(s, o))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(tt, l2_k0, alpha=1.0, linewidth=1.8, color='C0')

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(0, max(l2_k0))
    ax1.set_xlim(0, 1.01)
    # plt.yticks(np.arange(0, max(l2_k0), 0.2), fontsize=7)
    # plt.xticks(np.arange(0, 1801, 600), fontsize=7)

    plt.grid()
    ax1.set_box_aspect(1)
    plt.savefig(f"{folder_path}figures/L2_1observer.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_8obs_tf(multi_obs):
    a = np.load(f'{folder_path}pinns/weights_lambda_200.npy', allow_pickle=True)
    y = a[1:]
    w_f = y[:, -1]
    # x = np.linspace(0, 1, 101)
    # e = np.vstack((np.ones_like(x), np.ones_like(x))).T
    # sup_theta = ss.predict(e).reshape(x.shape)

    # fl = np.full_like(x, -1)
    # t = np.ones_like(x)
    # XS = np.vstack((x, t)).T
    # XO = np.vstack((x, sup_theta, fl, t)).T

    XO, y_true = gen_obsdata()

    XOf = XO[XO[:, 4]==1]
    x = XOf[:, 0:1]

    sf = y_true[-len(XOf):, :]
    o0f = multi_obs[0].predict(XOf)
    o1f = multi_obs[1].predict(XOf)
    o2f = multi_obs[2].predict(XOf)
    o3f = multi_obs[3].predict(XOf)
    o4f = multi_obs[4].predict(XOf)
    o5f = multi_obs[5].predict(XOf)
    o6f = multi_obs[6].predict(XOf)
    o7f = multi_obs[7].predict(XOf)
    o8f = w_f[0] * o0f + w_f[1] * o1f + w_f[2] * o2f + w_f[3] * o3f + w_f[4] * o4f + w_f[5] * o5f + w_f[6] * o6f + w_f[
        7] * o7f

    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', markersize=8, markevery=4, markeredgewidth=1.5, color='C0',
             label="System")
    ax2.plot(x, o0f, alpha=1.0, linewidth=1.8, color='C3', label="Observer1")
    ax2.plot(x, o1f, alpha=1.0, linewidth=1.8, color='lime', label="Observer2")
    ax2.plot(x, o2f, alpha=1.0, linewidth=1.8, color='blue', label="Observer3")
    ax2.plot(x, o3f, alpha=1.0, linewidth=1.8, color='purple', label="Observer4")
    ax2.plot(x, o4f, alpha=1.0, linestyle=(0, (5, 2)), linewidth=2.2, color='aqua', label="Observer5")
    ax2.plot(x, o5f, alpha=1.0, linestyle="dashdot", linewidth=1.8, color='lightskyblue', label="Observer6")
    ax2.plot(x, o6f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='darkred', label="Observer7")
    ax2.plot(x, o7f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='k', label="Observer8")
    ax2.plot(x, o8f, linestyle='None', marker="X", markersize=7, markevery=5, color='gold', label="MM adaptive observer")

    # ax2.set_ylim(39.5, 42)
    # ax2.set_ylim(0, 1)
    # ax2.set_xlim(0.665, 0.705)
    ax2.set_xlim(0, 1)
    # plt.yticks(np.arange(39.5, 42, 0.2))
    # plt.yticks(np.arange(0, 1.01, 0.1))
    # plt.xticks(np.arange(0.665, 0.706, 0.005))
    plt.xticks(np.arange(0, 1.01, 0.1))
    ax2.legend()
    ax2.set_ylabel(ylabel=r"Temperature")
    ax2.set_xlabel(xlabel=r"Distance $x$")
    ax2.set_title(r"Solutions at $\tau=t_f$", weight='semibold')

    plt.grid()

    # plt.savefig(f"{folder_path}figures/tf_8observers_zoomed.png",
    #            dpi=150, bbox_inches='tight')
    plt.savefig(f"{folder_path}figures/tf_8observers.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_8obs_l2(multi_obs):
    # x = np.linspace(0, 1, 100)
    # fl = np.full_like(x, -1)
    a = np.load(f'{folder_path}pinns/weights_lambda_200.npy', allow_pickle=True)

    """Plotting l2 error over time"""
    l2_k0 = []
    XO, y_true = gen_obsdata()
    tot = np.hstack((XO, y_true))
    tt = np.unique(tot[:, 4:5])

    for te in range(len(tt)):

        XOt = tot[tot[:, 4]==tt[te]]
        sf = XOt[:, 5:]
        o0f = multi_obs[0].predict(XOt[:, :5])
        o1f = multi_obs[1].predict(XOt[:, :5])
        o2f = multi_obs[2].predict(XOt[:, :5])
        o3f = multi_obs[3].predict(XOt[:, :5])
        o4f = multi_obs[4].predict(XOt[:, :5])
        o5f = multi_obs[5].predict(XOt[:, :5])
        o6f = multi_obs[6].predict(XOt[:, :5])
        o7f = multi_obs[7].predict(XOt[:, :5])
        o = a[1, te] * o0f + a[2, te] * o1f + a[3, te] * o2f + a[4, te] * o3f + a[5, te] * o4f + a[6, te] * o5f + \
            a[7, te] * o6f + a[8, te] * o7f

        l2_k0.append(dde.metrics.l2_relative_error(sf, o))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(tt, l2_k0, alpha=1.0, linewidth=1.8, color='C0')

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlim(0, 1.01)
    plt.yticks(fontsize=7)

    plt.grid()
    ax1.set_box_aspect(1)
    plt.savefig(f"{folder_path}figures/L2_8observers.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_hpo(ppre, ppost):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    x, matlab = gen_testdata()

    ll = int(np.sqrt(len(matlab)))

    pre = ppre.predict(x)
    post = ppost.predict(x)

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['Starting config', 'After HPO', 'Error w.r.t. MATLAB']

    # Define surfaces for each subplot
    surfaces = [
        [pre.reshape(ll, ll), post.reshape(ll, ll),
         np.abs(post - matlab).reshape(ll, ll)]
    ]

    # Create a grid of subplots
    grid = plt.GridSpec(1, 3)

    # Iterate over columns to add subplots
    for col in range(3):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        configure_subplot(ax, surfaces[0][col])

        # Set column titles
        ax.set_title(col_titles[col], fontsize=8, y=.96, weight='semibold')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15)

    # Save and show plot
    plt.savefig(f"{folder_path}figures/hpo_3d.png", dpi=300, bbox_inches='tight')
    plt.show()


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic):
    global ITERATION, gp_seed, dde_seed, output_path
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic]
    dde.config.set_random_seed(dde_seed)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="super-hpo-obs",

        # track hyperparameters and run metadata
        config={
            "learning rate": learning_rate,
            "num_dense_layers": num_dense_layers,
            "num_dense_nodes": num_dense_nodes,
            "activation": activation,
            "initialization": initialization,
            "w_domain": w_domain,
            "w_bcl": w_bcl,
            "w_bcr": w_bcr,
            "w_ic": w_ic        
        }
    )



    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print("initialization:", initialization)
    print("w_domain:", w_domain)
    print("w_bcl:", w_bcl)
    print("w_bcr:", w_bcr)
    print("w_ic:", w_ic)
    print()

    start_time = time.time()

    # Create the neural network with these hyper-parameters.
    mm = create_observer(config, 2.3)
    # possibility to change where we save
    error = train2_model(mm, ITERATION)
    # print(accuracy, 'accuracy is')

    if np.isnan(error):
        error = 10 ** 5

    end_time = time.time()
    time_spent = end_time - start_time

    # Store the configuration and error in a DataFrame
    data = {
        "Learning Rate": learning_rate,
        "Num Dense Layers": num_dense_layers,
        "Num Dense Nodes": num_dense_nodes,
        "Activation": activation,
        "Initialization": initialization,
        "W_domain": w_domain,
        "W_bcl": w_bcl,
        "W_bcr": w_bcr,
        "W_ic": w_ic,
        "Error": error,
        "Time Spent": time_spent
    }
    df = pd.DataFrame(data, index=[ITERATION])

    file_path = f"{output_path}hpo_results.csv"

    if not os.path.isfile(file_path):
        # Create a new CSV file with the header
        df.to_csv(file_path, index=False)
    else:
        # Append the DataFrame to the CSV file
        df.to_csv(file_path, mode='a', header=False, index=False)

    wandb.log({"err": error})
    wandb.finish()

    ITERATION += 1
    return error


def hpo(default_parameters):
    global gp_seed, dde_seed, output_path
    dde.config.set_random_seed(dde_seed)

    search_result = gp_minimize(
        func=fitness,
        dimensions=dimensions,
        acq_func="gp_hedge",  # Probability Improvement.
        n_calls=n_calls,
        x0=default_parameters,
        random_state=gp_seed,
    )

    print(search_result.x)

    # Plot convergence and save the figure
    plt.figure()
    plot_convergence(search_result)
    plt.title('Convergence Plot')
    plt.savefig(f"{output_path}figures/plot_conv.png",
            dpi=300, bbox_inches='tight')
    plt.show()

    # Plot objective and save the figure
    plt.figure()
    plot_objective(search_result, show_points=True, size=3.8)
    plt.savefig(f"{output_path}figures/plot_obj.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # Load the CSV file into a pandas DataFrame
    csv_file = f'{output_path}hpo_results.csv'

    return search_result.x


def mu(o, tau):
    XO, y_true = gen_obsdata()
    instants = np.unique(XO[:, 4:5])
    tot = np.hstack((XO, y_true))
    XO_all = tot[tot[:, 0]==np.max(tot[:, 0])]
    
    y1 = XO_all[:, 1:2].reshape(len(instants),)
    f1 = interp1d(instants, y1, kind='cubic')

    y2 = XO_all[:, 3:4].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='cubic')

    y3 = XO_all[:, 5:].reshape(len(instants),)
    f3 = interp1d(instants, y3, kind='cubic')

    XOt = np.hstack((np.max(tot[:, 0]), f1(tau), f2(tau), f3(tau), tau))
    th = f3(tau)
    muu = []
    for el in o:
        oss = el.predict(XOt)
        scrt = np.abs(oss-th)
        muu.append(scrt)
    muu = np.array(muu).reshape(len(muu),)

    return muu


def plot_weights(x, t, lam):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['C3', 'lime', 'blue', 'purple', 'aqua', 'lightskyblue', 'darkred', 'k']

    for i in range(x.shape[0]):
        plt.plot(1800 * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")

    ax1.set_xlim(0, 1800)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(f"Dynamic weights, $\lambda={lam}$", weight='semibold')
    plt.grid()
    plt.savefig(f"{folder_path}figures/weights_lam_{lam}.png", dpi=150, bbox_inches='tight')

    plt.show()




inizia()

