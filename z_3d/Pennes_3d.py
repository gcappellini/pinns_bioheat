"""Backend supported: tensorflow.compat.v1, paddle"""
import deepxde as dde
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dde_seed = 101

epochs, learning_rate = 10000, 1e-3
num_dense_layers, num_dense_nodes, activation, initialization = 4, 20, "tanh", "Glorot uniform"

# General parameters
L0, TM, Ta, tauf, qmet = 0.05, 45, 37, 1800, 4200

# Tissue parameters
rho, c, k_eff, W_min, W_avg, W_max, cb = 888, 2387, 1.2, 0.36, 0.54, 0.72, 3825           # fat
# rho, c, k_eff, W_min, W_avg, W_max, cb = 1050, 3639, 5, 0.45, 2.3, 4, 3825           # muscle

dT = TM - Ta
alfa = rho * c / k_eff

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)


# Backend tensorflow.compat.v1
def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    dtheta_yy = dde.grad.hessian(theta, x, i=0, j=1)
    dtheta_zz = dde.grad.hessian(theta, x, i=0, j=2)
    return a1 * dtheta_tau - dtheta_xx - dtheta_yy - dtheta_zz + a2 * W_avg * theta


def func(x):
    return 0

geom = dde.geometry.Sphere([0, 0, 0], 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)

ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=256,
    num_boundary=64,
    num_initial=32,
)

net = dde.nn.FNN([4] + [num_dense_nodes] * num_dense_layers + [1],
                 activation,
                 initialization)


model = dde.Model(data, net)
model.compile("adam", lr=learning_rate)
losshistory, train_state = model.train(iterations=epochs)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

X = geom.random_points(1000)
t = np.linspace(0, 1, num=100)
coord = np.vstack((X, t)).T
y_pred = model.predict(coord)
# print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
