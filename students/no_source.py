import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dde.config.set_random_seed(1)

learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = [0.001, 1, 30, "elu", "Glorot normal"]
w_domain, w_bcl, w_bcr, w_ic = [1, 1, 1, 1]

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
epochs = 20000
eee = 0.005

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


def source(s):
    return qmet #put here the SAR term

def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)


    return a1 * dtheta_tau - dtheta_xx + a2 * W_avg * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)  #temperature in x=0 must be 0
bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: -x[:, 1:], boundary_1)  #derivative of temperature (i.e., flux) in x=1 must be equal to -t

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
losshistory, train_state = model.train(iterations=epochs, model_save_path=f"{folder_path}model/no_source.ckpt")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Prediction grid
x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)

grid = np.vstack((np.ravel(X), np.ravel(T))).T
y_pred = model.predict(grid)
Y = np.reshape(y_pred, X.shape)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel("X")
ax1.set_ylabel("t")
ax1.set_zlabel("Temperature")

ax1.view_init(20, -120)

ax1.plot_surface(X, T, Y, cmap='inferno', alpha=.8)

plt.tight_layout()
plt.savefig(f"{folder_path}figures/no_source.png", dpi=300, bbox_inches='tight')
plt.show()



