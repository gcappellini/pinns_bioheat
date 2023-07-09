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

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin


dde_seed = 376

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
output_path = f"{folder_path}final/"

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


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def create_system(config):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic, end_time = config
    
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, end_time)
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

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic, end_time = config

    xmin = [0, -1]
    xmax = [1, 0]
    geom = dde.geometry.Rectangle(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, end_time)
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


def train_model(model, name):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{folder_path}model/{name}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{name}_loss",
                 train_fname=f"{name}_train", test_fname=f"{name}_test",
                 output_dir=f"{folder_path}history")
    return model


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
    plt.savefig(f"{folder_path}figures/system_3d.png", dpi=300, bbox_inches='tight')
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








