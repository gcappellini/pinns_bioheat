import torch
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


current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
epochs = 20000

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

# Considero il caso flusso costante
def gen_testdata():
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_system.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, -0.8*t, t)).T
    y = exact.flatten()[:, None]
    return X, y

def source(s):
    return qmet + beta*torch.exp(-cc*L0*(X0-s))*p

def pde(x, theta):

    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=2)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

    return a1 * dtheta_tau - dtheta_xx + a2 * W_avg * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def create_model(config):

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, start_flux, end_flux, end_time, w_domain, w_bcl, \
    w_bcr, w_ic = config

    geom = dde.geometry.Rectangle([0, start_flux], [1, end_flux])
    timedomain = dde.geometry.TimeDomain(0, end_time)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: x[:, 1:2], boundary_1)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_0, bc_1, ic],
        num_domain=2560,
        num_boundary=100,
        num_initial=160,
    )

    net = dde.nn.FNN(
        [3] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]
    model = dde.Model(data, net)
    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def restore_model(model):
    model.restore(f"{folder_path}model/aa.ckpt-{epochs}.pt", verbose=0)
    return model


def train_model(model):
    nn="aa"

    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{folder_path}model/{nn}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{nn}_loss",
                 train_fname=f"{nn}_train", test_fname=f"{nn}_test",
                 output_dir=f"{folder_path}history")
    
    train = np.array(losshistory.loss_train).sum(axis=1).ravel()

    error = train.min()
    return error