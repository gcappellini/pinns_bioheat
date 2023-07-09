from __future__ import print_function
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L0 = 0.05
TM = 45
Ta = 37
tauf = 1800

rho = 1050
c = 3639
k_eff = 5
alfa = rho * c / k_eff

W_avg = 2.3
W_min = 0.45
W_max = 4
cb = 3825

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff

q0 = 16
dT = TM - Ta
q0_ad = q0/dT

"""Implementing the system"""
N, sigma, lr, L_1, sharp = 20, "tanh", 1e-03, 3, 20000
dde.config.set_random_seed(101)

weights = 100
def pde_s(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

    return a1 * dtheta_tau - dtheta_xx + a2 * theta * W_avg


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: q0_ad, boundary_1)


def func(x):
    return q0_ad * (x[:, 0:1] ** 4) / 4 + 15 * (((x[:, 0:1] - 1) ** 2) * x[:, 0:1])/dT


ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

loss_weights = [1, 1, weights]
data = dde.data.TimePDE(
    geomtime,
    pde_s,
    [bc_1, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)

net = dde.nn.FNN([2] + [N] * L_1 + [1], sigma, "Glorot normal")


def output_transform(x, y):
    return x[:, 0:1] * y


net.apply_output_transform(output_transform)

sys = dde.Model(data, net)

# sys.compile("adam", lr=lr, loss_weights=loss_weights)
# losshistory, train_state = sys.train(iterations=sharp, model_save_path="model/new_sys.ckpt")
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)


sys.compile("adam", lr=lr, loss_weights=loss_weights)
sys.restore("model/new_sys.ckpt-20000.pt", verbose=0)


def y2(t):
    t = torch.tensor(t)
    one = torch.ones_like(t)
    Xp = torch.vstack((one, t)).T
    return torch.tensor(sys.predict(Xp))

def old_fun(x):
    e = np.array(sys.predict(x)).reshape(len(x), )
    return e

