import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd

f = pd.read_csv('RISULTATI_SAR.txt')
f1 = f.to_numpy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dde.config.set_random_seed(1)


learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = [0.001, 1, 30, "elu", "Glorot normal"]

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


# Muscle tissue parameters
rho = 1050
c = 3639
K = 0.2
k_eff = 5
alfa = rho * c / k_eff
# k_eff = k*(1+alfa*omegab)

W_avg = 2.3
W_min = 0.45
W_max = 4
cb = 3825
qmet=4200

dT = TM - Ta

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)

def source(s):
    p = 150/(1.97*10**-3)
    return qmet + tf.math.exp(-16*0.05*(0.09-s))*p


def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W_avg * theta - a3 * source(x[:,0:1])


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def gen_traindata():
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    Ca = np.reshape(f1, (-1, 1))
    return np.hstack((X, T)), Ca

observe_x, Ca= gen_traindata()
observe_y = dde.icbc.PointSetBC(observe_x, Ca, component=0)

data = dde.data.TimePDE(
    geomtime, pde, [observe_y] , num_domain=2560, num_boundary=100, anchors= observe_x ,num_initial=160)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1],
    activation,
    initialization,
)
model = dde.Model(data, net)

model.compile("adam", lr=learning_rate, loss_weights = [1,10000])
losshistory, train_state = model.train(iterations=epochs)


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
ax1.set_ylabel("tempo")
ax1.set_zlabel("Temperature")

ax1.view_init(20, -120)

ax1.plot_surface(X, T, Y, cmap='inferno',alpha=.8)
plt.savefig(f"{folder_path}figures/source_SAR.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

error= dde.metrics.l2_relative_error(f1, Y)
print(error)

diff = f1- Y
diff1 = np.abs(diff)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel("X")
ax1.set_ylabel("tempo")
ax1.set_zlabel("Errore")

ax1.view_init(20, -120)
ax1.plot_surface(X, T, diff1, cmap='inferno',alpha=.8)
plt.tight_layout()
plt.show()
