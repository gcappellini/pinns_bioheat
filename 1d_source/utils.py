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

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin


gp_seed = None
dde_seed = 129
ITERATION = 0

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

epochs = 20000
n_calls = 500

x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)
xs = np.vstack((np.ravel(X), np.ravel(T))).T
xsup = np.vstack((np.ones_like(np.ravel(T)), np.ravel(T))).T


# General parameters
L0 = 0.05
TM = 45
Ta = 37
tauf = 1800
qmet = 4200

# Fat tissue parameters
rho = 940
c = 2500
K = 0.2
k_eff = 5
alfa = rho * c / k_eff
# k_eff = k*(1+alfa*omegab)

W_avg = 0.54
W_min = 0.36
W_max = 0.72
cb = 3825

dT = TM - Ta

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)

# Antenna parameters
beta = 1
cc = 16
X0 = 0.09
p = 150/(5.75e-3)

# Observer parameters
k = 4


# HPO setting
n_calls = 100
dim_learning_rate = Real(low=1e-5, high=5e-1, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=4, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=200, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"],
                             name="activation")
dim_initialization = Categorical(categories=["Glorot normal", "Glorot uniform", "He normal", "He uniform"],
                             name="initialization")
dim_weight_ic = Integer(low=1, high=100, name="weight_ic")
dim_weight_bcl = Integer(low=1, high=100, name="weight_bcl")
dim_weight_bcr = Integer(low=1, high=100, name="weight_bcr")
dim_weight_domain = Integer(low=1, high=100, name="weight_domain")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization,
    dim_weight_ic,
    dim_weight_bcl,
    dim_weight_bcr,
    dim_weight_domain
]


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
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_system.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y


def source(s):
    return qmet + beta*torch.exp(-cc*L0*(X0-s))*p


def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W_avg * theta - a3 * source(x[:, 0:1])


def pde_m(x, theta, W):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def create_system(config):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config
    
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: -0.8*x[:, 1:], boundary_1)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, pde, [bc_0, bc_1, ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )
    model = dde.Model(data, net)

    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def ic_obs(x):
    return x[:, 0:1] * (6/5 - x[:, 0:1])


def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    return dtheta_x + 0.8 * x[:, 2:] + k * theta - k * x[:, 1:2]

def create_observer(config):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    xmin = [0, -1]
    xmax = [1, 0]
    geom = dde.geometry.Rectangle(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, pde, [bc_0, bc_1, ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [3] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )
    model = dde.Model(data, net)

    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def create_multi_obs(config, W):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    xmin = [0, -1]
    xmax = [1, 0]
    geom = dde.geometry.Rectangle(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_m(x, theta, W), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100,
        num_initial=160
    )

    net = dde.nn.FNN(
        [3] + [num_dense_nodes] * num_dense_layers + [1],
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

    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{output_path}model/{name}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{name}_loss",
                 train_fname=f"{name}_train", test_fname=f"{name}_test",
                 output_dir=f"{output_path}history")
    return model
    # X, y_true = gen_testdata()
    # y_pred = model.predict(X)
    # er = dde.metrics.l2_relative_error(y_true, y_pred)
    # return er


def restore_model(model, name):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    model.restore(f"{folder_path}model/{name}.ckpt-{epochs}.pt", verbose=0)
    return model


def configure_subplot(ax, surface):
    ax.plot_surface(X, 1800 * T, surface, cmap='inferno', alpha=.8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)


def plot_3d(p, lbl):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    x, matlab = gen_testdata()

    ll = int(np.sqrt(len(matlab)))

    pinns = p.predict(x)

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['MATLAB', 'PINNs', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [matlab.reshape(ll, ll), pinns.reshape(ll, ll),
         np.abs(pinns - matlab).reshape(ll, ll)]
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
    plt.savefig(f"{folder_path}figures/{lbl}_3d.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_obs(s, o):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    sys = s.predict(xs)
    xo = np.vstack((np.ravel(X), s.predict(xsup).reshape(len(np.ravel(X)),), np.ravel(T))).T
    obs = o.predict(xo)

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['System', 'Observer', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [sys.reshape(X.shape), obs.reshape(X.shape),
         np.abs(sys - obs).reshape(X.shape)]
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
    plt.savefig(f"{folder_path}figures/observer_3d.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_obs(s, o, ww):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    sys = s.predict(xs)
    xo = np.vstack((np.ravel(X), s.predict(xsup).reshape(len(np.ravel(X)),), np.ravel(T))).T
    obs = o.predict(xo)

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['System', 'Observer', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [sys.reshape(X.shape), obs.reshape(X.shape),
         np.abs(sys - obs).reshape(X.shape)]
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
    plt.savefig(f"{folder_path}figures/{ww}_3d.png", dpi=300, bbox_inches='tight')
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
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, weight_ic, weight_bcl, weight_bcr, weight_domain):
    global ITERATION, gp_seed, dde_seed, output_path
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, weight_ic, weight_bcl, weight_bcr, weight_domain]
    dde.config.set_random_seed(dde_seed)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="hpo-syst-source",

        # track hyperparameters and run metadata
        config={
            "learning rate": learning_rate,
            "num_dense_layers": num_dense_layers,
            "num_dense_nodes": num_dense_nodes,
            "activation": activation,
            "initialization": initialization,
            "weight_ic": weight_ic,
            "weight_bcl": weight_bcl,
            "weight_bcr": weight_bcr,
            "weight_domain": weight_domain        
        }
    )



    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print("initialization:", initialization)
    print("weight_ic:", weight_ic)
    print("weight_bcl:", weight_bcl)
    print("weight_bcr:", weight_bcr)
    print("weight_domain:", weight_domain)
    print()

    start_time = time.time()

    # Create the neural network with these hyper-parameters.
    mm = create_system(config)
    # possibility to change where we save
    error = train_model(mm, ITERATION)
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
        "Weight_ic": weight_ic,
        "Weight_bcl": weight_bcl,
        "Weight_bcr": weight_bcr,
        "Weight_domain": weight_domain,
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
        acq_func="PI",  # Probability Improvement.
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


def mu(s, o, tau):
    xs = np.vstack((np.ones_like(tau), tau)).T
    th = s.predict(xs)
    xo = np.vstack((np.ones_like(tau), th, tau)).T
    muu = []
    for el in o:
        oss = el.predict(xo)
        scrt = np.abs(oss-th)
        muu.append(8*scrt)
    muu = np.array(muu).reshape(len(muu),)
    return muu


def f(t, p):
    return 0



