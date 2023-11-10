import os
import deepxde as dde
from deepxde.geometry.geometry_3d import Sphere, Cuboid
from deepxde.geometry.csg import CSGDifference, CSGUnion, CSGIntersection
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import pandas as pd
from openpyxl import load_workbook
import random
import imgkit
import torch
import wandb
from scipy import integrate
from scipy.interpolate import interp1d
import time

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
L0, TM, Ta, tauf, qmet, max_t = 0.2, 30, 20, 1800, 0, 5

# Phantom parameters
rho, c, k = 1000, 4180, 0.6

# Convective heat transfer 
# (https://help.solidworks.com/2022/english/SolidWorks/cworks/c_Convection.htm#jmd1450460168782)
# (https://help.solidworks.com/2022/english/SolidWorks/cworks/c_Thermal_Contact_Resistance.htm)
# h_up, h_down = 10, 45000
h_up, h_down = 10, 0.01

dT = TM - Ta
alfa = rho * c / k

a1 = (alfa * (L0 ** 2)) / tauf
a3 = (L0 ** 2) / (k * dT)

# Antenna parameters
# beta, aa, bb, cc, p = 1, 64, 111, 16, 150
beta, aa, bb, cc, p = 100, 640, 1111, 16, 750
X0, Y0, Z0 = 0.5, 0.5, 0.994

# Observer parameter
kk = 10

# Define the dimensions of the cylinder
radius = 0.7  # Radius of the cylinder
height = 1.0  # Height of the cylinder

# Create the sphere for the curved part of the cylinder
sphere = Sphere(center=(0.5, 0.5, 0.5), radius=radius)

# Create the rectangle for the flat part of the cylinder
rectangle = Cuboid(xmin=[0, 0, 0.5], xmax=[height, height, 1.0])
rectangle2 = Cuboid(xmin=[0, 0, 0], xmax=[1.0, 1.0, 0.5])

# Perform a union operation to combine the sphere and rectangle into a cylinder
emysphere = CSGIntersection(sphere, rectangle)
geom = CSGUnion(emysphere, rectangle2)

timedomain = dde.geometry.TimeDomain(0, max_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# HPO setting
n_calls = 500
dim_learning_rate = Real(low=0.000012774471609203795, high=0.21788060459464648, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=8, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=7, high=384, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"],
                             name="activation")
# ["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"]
dim_initialization = Categorical(categories=["Glorot uniform", "He normal", "He uniform"],
                             name="initialization")
dim_w_domain = Integer(low=0, high=250, name="w_domain")
dim_w_bc0 = Integer(low=0, high=250, name="w_bc0")
dim_w_bc1 = Integer(low=0, high=198, name="w_bc1")
dim_w_ic = Integer(low=0, high=182, name="w_ic")


dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization,
    dim_w_ic,
    dim_w_bc0,
    dim_w_bc1,
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


def step(x):
    # Create a tensor filled with 1s of the same shape as x
    ones = torch.ones_like(x)
    
    # Create a tensor filled with 0s of the same shape as x
    zeros = torch.zeros_like(x)
    
    # Use torch.where to perform an element-wise conditional operation
    result = torch.where((x > 0.01) & (x < 1), ones, zeros)
    
    return result


def step2(x):
    
    # Create a tensor filled with 0s of the same shape as x
    zeros = torch.zeros_like(x)
    
    # Use torch.where to perform an element-wise conditional operation
    result = torch.where(x > 0, x, zeros)
    
    return result


def source(s):
    ss = qmet + beta*step(s[:, 3:])*p*torch.exp(-L0**2*(aa*(s[:, 0:1]-X0)**2+bb*(s[:, 1:2]-Y0)**2+cc*(Z0/s[:, 2:3])))
    return ss.reshape(len(s), 1)
    # return 0

def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    dtheta_yy = dde.grad.hessian(theta, x, i=1, j=1)
    dtheta_zz = dde.grad.hessian(theta, x, i=2, j=2)
    return a1 * dtheta_tau - dtheta_xx - dtheta_yy - dtheta_zz - a3 * source(x)


def boundary_d(x, on_boundary):
    return on_boundary and np.isclose(x[2], 0)


def boundary_u(x, on_boundary):
    return on_boundary and np.greater(x[2], 0)


def create_system(config):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcd, w_bcu, w_ic = config

    # bc_d = dde.icbc.NeumannBC(geomtime, lambda X: 0, boundary_d)
    # bc_u = dde.icbc.NeumannBC(geomtime, lambda X: -.1, boundary_u)

    bc_d = dde.icbc.RobinBC(geomtime, lambda X, y: - (h_down/k)*step2(y), boundary_d)
    bc_u = dde.icbc.RobinBC(geomtime, lambda X, y: - (h_up/k)*step2(y), boundary_u)

    ic = dde.icbc.IC(
        geomtime,
        lambda x: 0,
        lambda _, on_initial: on_initial,
    )

    data = dde.data.TimePDE(
        geomtime, pde, [bc_d, bc_u, ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [4] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    model = dde.Model(data, net)
    loss_weights = [w_domain, w_bcd, w_bcu, w_ic]

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


def restore_model(model, name):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    model.restore(f"{folder_path}model/{name}.ckpt-{epochs}.pt", verbose=0)
    return model

def plot(o, time_inst):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    X0 = geomtime.random_points(5000)
    add_time = np.full((X0.shape[0], 1), time_inst)
    X0_without_last_column = X0[:, :-1]
    X0t = np.hstack((X0_without_last_column, add_time))
    sys = o.predict(X0t)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the x, y, and z coordinates from X0t
    x_coords = X0t[:, 0]
    y_coords = X0t[:, 1]
    z_coords = X0t[:, 2]

    # Plot the output 'sys' at the 3D coordinates
    sc = ax.scatter(x_coords, y_coords, z_coords, c=sys, cmap='viridis', vmax=1, vmin=0)#, vmax=np.max(sys), vmin=np.min(sys))

    # Add a color bar to the plot
    plt.colorbar(sc)

    # Set labels for the axes
    ax.set_title(f'System at t={time_inst}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_box_aspect([1, 1, 1])

    # Show the plot
    plt.savefig(f"{folder_path}figures/plot_t={time_inst}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_needle(model):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    size = 100
    z = np.linspace(0, 1, num=size)
    t = np.linspace(0, max_t, num=size)
    ZZ, TT = np.meshgrid(z, t)
    x_coord, y_coord = np.full_like(np.ravel(ZZ), 0.5), np.full_like(np.ravel(ZZ), 0.5)

    X_needle = np.vstack((x_coord, y_coord, np.ravel(ZZ), np.ravel(TT))).T
    needle = model.predict(X_needle)
    N = needle.reshape(size,size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 3D surface plot
    surf = ax.plot_surface(ZZ, TT, N, cmap='inferno', alpha=0.8)

    # Add a colorbar
    # cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)


    # Set labels for the axes
    ax.set_title(f'Observation needle')
    ax.set_xlabel('Z-axis')
    ax.set_ylabel('time')
    ax.set_zlabel('temperature')
    ax.set_box_aspect([1, 1, 1])

    # Show the plot
    plt.savefig(f"{folder_path}figures/plot_needle.png", dpi=300, bbox_inches='tight')
    plt.show()


def pde_obs(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=4)
    dtheta_zz = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_zz


def ic_obs(x):
    y1 = x[0, 1:2]
    y2 = x[0, 2:3]
    y3 = x[0, 3:4]
    b1 =  6# arbitrary parameter
    # a2 = 0
    # e = y1 + (y3 - 2*a2 + k*(y2-y1-a2))*x + a2*x**2
    e = (y3 + kk * (y2 - y1))/(b1 * np.cos(b1) + kk * np.sin(b1))* np.sin(b1*x) + y1
    return e[:, 0:1]


def bc0_obs(x, theta, X):
    return x[:, 1:2] - theta


def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    return dtheta_x - x[:, 3:4] - kk * (x[:, 2:3] - theta) 


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def create_observer(config):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bc0, w_bc1, w_ic = config

    xmin = [0, 0, 0, -5]
    xmax = [1, 1, 1, +5]
    geom = dde.geometry.Hypercube(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(1, max_t)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.OperatorBC(geomtime, bc0_obs, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_obs(x, theta), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [5] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    model = dde.Model(data, net)
    loss_weights = [w_domain, w_bc0, w_bc1, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def gen_obscoord(data):
    z = data[:, 2:3]
    z_end = np.min(z)
    theta_end = data[data[:, 2] == z_end]

    unique_z = np.unique(z)
    sorted_unique_z = np.sort(unique_z)[::-1]
    z_sup = sorted_unique_z[1]
    theta_sup = data[data[:, 2] == z_sup]

    z_max = np.max(z)
    theta_extra = data[data[:, 2] == z_max]
    fl = (theta_extra[:, 4:] - theta_sup[:, 4:])/(z_max - z_sup)

    mask = data[:, 2] != z_max
    data = data[mask, :]
    z, t, exact = data[:, 2:3], data[:, 3:4], data[:, 4:]
    y = exact.flatten()[:, None]
    num_rows = data.shape[0]
    new_columns = np.zeros((num_rows, 3))
    X = np.hstack((z, new_columns, t))

    unique_t = np.unique(t)
    for el in range(len(unique_t)):
        a = X[X[:, 4] == unique_t[el]]
        a[:, 1] = theta_end[el, 4:]
        a[:, 2] = theta_sup[el, 4:]
        a[:, 3] = fl[el]
        X[X[:, 4] == unique_t[el]] = a

    return X, y


def configure_subplot(ax, XS, surface):
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))
    X = XS[:, 0:1].reshape(le, la)
    T = XS[:, 1:].reshape(le, la)

    ax.plot_surface(X, T, surface, cmap='inferno', alpha=.8)
    ax.set_xlabel('Z-axis')
    ax.set_ylabel('time')
    ax.set_zlabel('temperature')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)


def plot_1obs(s, o, n_guide):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    # Measurement linspace
    z_needle = np.linspace(0, 1, num=n_guide)
    t_meas = np.linspace(1, 5, num=100)
    ZZ, TT = np.meshgrid(z_needle, t_meas)
    xy_needle = np.full_like(np.ravel(ZZ), 0.5)
    coord_needle = np.vstack((xy_needle, xy_needle, np.ravel(ZZ), np.ravel(TT))).T
    meas = s.predict(coord_needle)

    # Observer prediction
    info_obs = np.hstack((coord_needle, meas))
    coord_obs, sys = gen_obscoord(info_obs)
    obs = o.predict(coord_obs)

    # Create 3D axes
    fig = plt.figure(figsize=(9, 4))

    # Define column titles
    col_titles = ['System', 'Observer', 'Error']

    XS = np.delete(coord_obs, [1, 2, 3], axis=1)
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


def plot_1obs_tf(s, o, n_guide):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    # Measurement linspace
    z_needle = np.linspace(0, 1, num=n_guide)
    t_meas = np.full_like(z_needle, max_t)
    xy_needle = np.full_like(z_needle, 0.5)
    coord_needle = np.vstack((xy_needle, xy_needle, z_needle, t_meas)).T
    meas = s.predict(coord_needle)

    # Observer prediction
    info_obs = np.hstack((coord_needle, meas))
    coord_obs, sf = gen_obscoord(info_obs)
    of = o.predict(coord_obs)
    z = coord_obs[:, 0:1]

    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot(z, sf, marker="o", mfc='none', color='C0', label="System")#, linestyle='None', markevery=2)
    ax2.plot(z, of, marker="x", color='C1', label="Observer")#, linestyle='None', markevery=3)

    ax2.legend()
    ax2.set_ylabel(ylabel=r"Temperature")
    ax2.set_xlabel(xlabel=r"Depth $z$")
    ax2.set_title(r"Solution at $t=t_f$", weight='semibold')

    plt.grid()

    plt.savefig(f"{folder_path}figures/tf_1observer.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_1obs_l2(s, o, n_guide):
    l2_k0 = []


    # Measurement linspace
    z_needle = np.linspace(0, 1, num=n_guide)
    t_meas = np.linspace(1, max_t, num=100)
    ZZ, TT = np.meshgrid(z_needle, t_meas)
    xy_needle = np.full_like(np.ravel(ZZ), 0.5)
    coord_needle = np.vstack((xy_needle, xy_needle, np.ravel(ZZ), np.ravel(TT))).T
    meas = s.predict(coord_needle)
    
    # Observer prediction
    info_obs = np.hstack((coord_needle, meas))
    coord_obs, sys = gen_obscoord(info_obs)

    tot = np.hstack((coord_obs, sys))
    tt = np.unique(tot[:, 4:5])
    for t in tt:
        XOt = tot[tot[:, 4]==t]
        ss = XOt[:, 5:]
        oo = o.predict(XOt[:, :5])
        l2_k0.append(dde.metrics.l2_relative_error(ss, oo))


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(tt, l2_k0, alpha=1.0, linewidth=1.8, color='C0')

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(0, max(l2_k0))
    # ax1.set_xlim(0, 1.01)
    # plt.yticks(np.arange(0, max(l2_k0), 0.2), fontsize=7)
    # plt.xticks(np.arange(0, 1801, 600), fontsize=7)

    plt.grid()
    ax1.set_box_aspect(1)
    plt.savefig(f"{folder_path}figures/l2_1observer.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def l2_penalty(s, o, n_guide):
    l2_k0 = []

    # Measurement linspace
    z_needle = np.linspace(0, 1, num=n_guide)
    t_meas = np.linspace(1, max_t, num=100)
    ZZ, TT = np.meshgrid(z_needle, t_meas)
    xy_needle = np.full_like(np.ravel(ZZ), 0.5)
    coord_needle = np.vstack((xy_needle, xy_needle, np.ravel(ZZ), np.ravel(TT))).T
    meas = s.predict(coord_needle)
    
    # Observer prediction
    info_obs = np.hstack((coord_needle, meas))
    coord_obs, sys = gen_obscoord(info_obs)

    tot = np.hstack((coord_obs, sys))
    tt = np.unique(tot[:, 4:5])
    for t in tt:
        XOt = tot[tot[:, 4]==t]
        ss = XOt[:, 5:]
        oo = o.predict(XOt[:, :5])
        l2_k0.append(t*dde.metrics.l2_relative_error(ss, oo))
    
    return np.linalg.norm(l2_k0)

confi = [0.0001046, 3, 165, "sigmoid", "He normal", 1, 1, 1, 10000]
ded = create_system(confi)
# o = utils2.train_model(d, "sys")
oeo = restore_model(ded, "sys")

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bc0, w_bc1, w_ic):
    global ITERATION, gp_seed, dde_seed, output_path
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bc0, w_bc1, w_ic]
    dde.config.set_random_seed(dde_seed)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="obs-simulation",

        # track hyperparameters and run metadata
        config={
            "learning rate": learning_rate,
            "num_dense_layers": num_dense_layers,
            "num_dense_nodes": num_dense_nodes,
            "activation": activation,
            "initialization": initialization,
            "w_domain": w_domain,
            "w_bc0": w_bc0,
            "w_bc1": w_bc1,
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
    print("w_bc0:", w_bc0)
    print("w_bc1:", w_bc1)
    print("w_ic:", w_ic)
    print()

    start_time = time.time()

    # Create the neural network with these hyper-parameters.
    mm = create_observer(config)
    nn = train_model(mm, f"obs_{ITERATION}")
    # possibility to change where we save
    error = l2_penalty(oeo, nn, 25)
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
        "W_bc0": w_bc0,
        "W_bc1": w_bc1,
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

    return search_result.x