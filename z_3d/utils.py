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
dde_seed = 101
ITERATION = 0

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

epochs = 20000

# General parameters
L0 = 0.05
TM = 45
Ta = 37
tauf = 1800

qmet = 4200

# Tissue parameters
# rho, c, k_eff, W_min, W_avg, W_max = 888, 2387, 1.2, 0.36, 0.54, 0.72           # fat
rho, c, k_eff, W_min, W_avg, W_max = 1050, 3639, 5, 0.45, 2.3, 4           # muscle
cb = 3825

dT = TM - Ta
alfa = rho * c / k_eff

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)


# Antenna parameters
beta = 1
aa = 640
bb = 1111
cc = 16
z0 = 0.004
p = 6e6

# Observer parameters
k = 4


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
    x = s[:, 0:1]
    y = s[:, 1:2]
    z = s[:, 2:3]
    return qmet + beta*torch.exp(-aa*x**2 - bb*(y)**2 -cc*L0*(z0-z))*p


def pde_s(x, theta, W):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    dtheta_yy = dde.grad.hessian(theta, x, i=0, j=1)
    dtheta_zz = dde.grad.hessian(theta, x, i=0, j=2)
    return a1 * dtheta_tau - dtheta_xx - dtheta_yy - dtheta_zz + a2 * W * theta - a3 * source(x)


def pde_m(x, theta, W):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)



def create_system(config, W):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config
    
    xmin = [0, 0, 0]
    xmax = [1, 1, 1]
    geom = dde.geometry.Cuboid(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    # bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: -1, boundary_1)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_s(x, theta, W), [ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [4] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )


    model = dde.Model(data, net)

    loss_weights = [w_domain, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def ic_obs(x):
    return x[:, 0:1] * (6/5 - x[:, 0:1])



def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    # return dtheta_x + x[:, 2:3] + k * theta - k * x[:, 1:2]
    return dtheta_x - x[:, 2:3] - k * (x[:, 1:2] - theta) 



def output_transform(x, y):
    return x[:, 0:1] * y



def create_observer(config, W):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    xmin = [0, 0, -2]
    xmax = [1, 1, 0]
    geom = dde.geometry.Cuboid(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_m(x, theta, W), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [4] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    # net.apply_output_transform(output_transform)

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


def sup_theta(e):
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    xsup = np.vstack((np.ones_like(np.ravel(T)), np.ravel(T))).T
    xx = np.vstack((np.ravel(X), np.ravel(T))).T
    a = e.predict(xsup).T
    b = e.predict(xx).T
    np.savez(f"{folder_path}pinns/sup_theta.npz", data=a)
    np.savez(f"{folder_path}pinns/theta.npz", data=b)
    # return a


def gen_obsdata():
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_observer.txt")
    supp_theta = np.load(f"{folder_path}pinns/sup_theta.npz")
    sup_theta = supp_theta['data']
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    fl = np.full_like(t, -1)
    X = np.vstack((x, sup_theta, fl, t)).T
    y = exact.flatten()[:, None]
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


def configure_subplot(ax, surface):
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    ax.plot_surface(X, 1800 * T, surface, cmap='inferno', alpha=.8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)


def plot_3d(p):
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
    plt.savefig(f"{folder_path}figures/cfr_sys.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_obs(o):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    XO, y_true = gen_obsdata()
    y_pred = o.predict(XO)

    ll = int(np.sqrt(len(y_true)))

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['MATLAB', 'PINNs', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [y_true.reshape(ll, ll), y_pred.reshape(ll, ll),
         np.abs(y_true - y_pred).reshape(ll, ll)]
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
    plt.savefig(f"{folder_path}figures/cfr_obs.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_1obs(s, o):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    XO, y_true = gen_obsdata()

    XS = np.delete(XO, [1, 2], axis=1)
    sys = s.predict(XS)
    obs = o.predict(XO)

    ll = int(np.sqrt(len(y_true)))

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['System', 'Observer', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [sys.reshape(ll, ll), obs.reshape(ll, ll),
         np.abs(sys - obs).reshape(ll, ll)]
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
    plt.savefig(f"{folder_path}figures/plot_1obs.png", dpi=300, bbox_inches='tight')
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


def mu(s, o, tau):
    xs = np.vstack((np.ones_like(tau), tau)).T
    th = s.predict(xs)
    fl = np.full_like(tau, -1)
    xo = np.vstack((np.ones_like(tau), th, fl, tau)).T
    muu = []
    for el in o:
        oss = el.predict(xo)
        scrt = np.abs(oss-th)
        muu.append(scrt)
    muu = np.array(muu).reshape(len(muu),)
    return muu


def plot_weights(x, t, lam):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['C3', 'lime', 'blue', 'purple', 'aqua', 'lightskyblue', 'darkred', 'k']

    for i in range(x.shape[0]):
        plt.plot(1800 * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i}$")

    ax1.set_xlim(0, 1800)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(f"Dynamic weights, $\lambda={lam}$", weight='semibold')
    plt.grid()
    plt.savefig(f"{folder_path}figures/weights_lam_{lam}.png", dpi=150, bbox_inches='tight')

    plt.show()


def plot_temperature_colormap(model, x, y, z, t):
    # Generate a grid for the entire cuboid (x, y, z)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a 4D array for (x, y, z, t) coordinates
    coords = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), t * np.ones_like(x_grid).ravel())).T
    
    # Predict temperature using the PINN model for the given time instant
    temperature = model.predict(coords)
    
    # Reshape temperature back to the original grid shape
    temperature_grid = temperature.reshape(x_grid.shape)
    
    # Plot the temperature distribution as a colormap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Choose a colormap that represents temperature well (e.g., 'hot')
    cmap = plt.cm.get_cmap('hot')
    
    # Create the colormap plot
    surf = ax.scatter(x_grid, y_grid, z_grid, c=temperature_grid, cmap=cmap, marker='o')
    
    # Add a colorbar for temperature visualization
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Temperature')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set the title
    ax.set_title(f'Temperature distribution at t = {t}')
    file_name = f"{folder_path}figures/3d_temperature_t_{t}.png"
    plt.savefig(file_name)
    # Show the plot
    plt.show()


