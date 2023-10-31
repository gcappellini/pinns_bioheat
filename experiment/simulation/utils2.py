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

epochs = 3000

# General parameters
L0, TM, Ta, tauf, qmet = 0.2, 30, 20, 1800, 0

# Phantom parameters
rho, c, k = 1000, 4180, 0.6

dT = TM - Ta
alfa = rho * c / k

a1 = (alfa * (L0 ** 2)) / tauf
a3 = (L0 ** 2) / (k * dT)

# Antenna parameters
# beta, aa, bb, cc, p = 1, 64, 111, 16, 150
beta, aa, bb, cc, p = 100, 640, 1111, 16, 150
X0, Y0, Z0 = 0.5, 0.5, 0.994

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

timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(sphere, timedomain)

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


def source(s):
    ss = qmet + beta*p*torch.exp(-L0**2*(aa*(s[:, 0:1]-X0)**2+bb*(s[:, 1:2]-Y0)**2+cc*(Z0/s[:, 2:3])))
    return ss.reshape(len(s), 1)
    # return qmet

def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    dtheta_yy = dde.grad.hessian(theta, x, i=1, j=1)
    dtheta_zz = dde.grad.hessian(theta, x, i=2, j=2)
    return a1 * dtheta_tau - dtheta_xx - dtheta_yy - dtheta_zz - a3 * source(x)


def boundary_d(x, on_boundary):
    return on_boundary and np.isclose(x[3], 0)


def boundary_u(x, on_boundary):
    return on_boundary and np.greater(x[3], 0)


def create_system(config):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcd, w_bcu, w_ic = config

    bc_d = dde.icbc.NeumannBC(geomtime, lambda X: -0.1, boundary_d)
    bc_u = dde.icbc.NeumannBC(geomtime, lambda X: -1, boundary_d)

    ic = dde.icbc.IC(
        geomtime,
        lambda x: 0,
        lambda _, on_initial: on_initial,
    )

    # data = dde.data.TimePDE(
    #     geomtime, pde, [bc_d, bc_u, ic], num_domain=2560, num_boundary=100, num_initial=160
    # )

    data = dde.data.TimePDE(
        geomtime, pde, [], num_domain=2560, num_boundary=100, num_initial=160
    )

    net = dde.nn.FNN(
        [4] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    model = dde.Model(data, net)
    # loss_weights = [w_domain, w_bcd, w_bcu, w_ic]
    loss_weights = [w_domain]

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

    X0 = geomtime.uniform_points(5000)
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
    sc = ax.scatter(x_coords, y_coords, z_coords, c=sys, cmap='viridis') # , marker='o')

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
