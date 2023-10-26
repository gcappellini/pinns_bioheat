"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = [0.001, 2, 30, "elu", "Glorot normal"]

# General parameters
L0, TM, Ta, tauf, qmet = 0.05, 45, 37, 1800, 4200

# Tissue parameters
# rho, c, k_eff, W_min, W_avg, W_max, cb = 888, 2387, 1.2, 0.36, 0.54, 0.72, 3825           # fat
rho, c, k_eff, W_min, W_avg, W_max, cb = 1050, 3639, 5, 0.45, 2.3, 4, 3825           # muscle

dT = TM - Ta
alfa = rho * c / k_eff

a1, a2, a3 = (alfa * (L0 ** 2)) / tauf, (L0 ** 2) * cb / k_eff, (L0 ** 2) / (k_eff * dT)

# Antenna parameters
beta, cc, X0, p = 1, 16, 0.09, 76142.131

k = 1


def source(s):
    return qmet + beta*torch.exp(-cc*L0*(X0-s))*p


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def gen_end_data():
    data = np.loadtxt(f"{folder_path}output_matlab_pde.txt")
    x, t = data[:, 0:1], data[:, 1:2]

    # Find the minimum value in x
    x_end = np.min(x)

    # Find the rows where x is equal to x_end
    theta_end = data[x[:, 0] == x_end]

    return theta_end[:, 0:2], theta_end[:, 2:]

def gen_sup_data():
    data = np.loadtxt(f"{folder_path}output_matlab_pde.txt")
    x, t = data[:, 0:1], data[:, 1:2]

    # Find the unique values of x and sort them in descending order
    unique_x = np.unique(x)
    sorted_unique_x = np.sort(unique_x)[::-1]
    x_sup = sorted_unique_x[1]

    # Find the rows where x is equal to x_sup
    theta_sup = data[x[:, 0] == x_sup]

    x_max = np.max(x)
    theta_extra = data[x[:, 0] == x_max]

    fl = (theta_extra[:, 2:] - theta_sup[:, 2:])/(x_max - x_sup)


    theta_sup = np.hstack((theta_sup, fl))
    theta_sup[:, 0] = 1
    return theta_sup


def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    # return dtheta_x + x[:, 2:3] + k * theta - k * x[:, 1:2]
    return dtheta_x - x[:, 2:3] - k * (x[:, 1:2] - theta)


def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W_avg * theta - a3 * source(x[:, 0:1])


def ic_obs(x):
    a2 = 3
    y2 = 0
    y3 = 0
    return x[:, 0:1] * (k*y2 + y3 - a2*(k+2))/(k+1) + a2 * x[:, 0:1]**2

# Computational geometry:
# geom = dde.geometry.Cuboid([0, 0, -5], [1, 1, 5])
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ob_x, ob_end = gen_end_data()
observe_end = dde.icbc.PointSetBC(ob_x, ob_end, component=0)

# Initial and boundary conditions:
ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)


# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_end],
    num_domain=2540,
    num_initial=160,
    num_test=2540,
)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Build and train the model:
model.compile("adam", lr=1e-3)
model.train(iterations=20000)
model.compile("L-BFGS")
losshistory, train_state = model.train(model_save_path=f"{folder_path}model/observer.ckpt")

# model.compile("L-BFGS")
# model.restore(f"{folder_path}model/observer.ckpt-35000.pt", verbose=0)

# Plot/print the results
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

y_pred = model.predict(X)
f = model.predict(X, operator=pde)

print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))


ll = int(np.sqrt(len(y_true)))

XC = X[:, 0:1].reshape(ll, ll)
T = X[:, 3:].reshape(ll, ll)
Y_pred = y_pred.reshape(ll, ll)
Y_true = y_true.reshape(ll, ll)
Y = np.abs(Y_true - Y_pred)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Plot Y_true
axs[0].plot_surface(XC, T, Y_true, cmap='inferno', alpha=.8)
axs[0].set_title("System")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("X")
axs[0].set_zlabel("Temperature")
axs[0].view_init(20, 20)

# Plot Y_pred
axs[1].plot_surface(XC, T, Y_pred, cmap='inferno', alpha=.8)
axs[1].set_title("Observer")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("X")
axs[1].set_zlabel("Temperature")
axs[1].view_init(20, 20)

# Plot Y
axs[2].plot_surface(XC, T, Y, cmap='inferno', alpha=.8)
axs[2].set_title("Error")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("X")
axs[2].set_zlabel("Temperature")
axs[2].view_init(20, 20)

plt.tight_layout()
plt.savefig(f"{folder_path}figures/obs.png", dpi=300, bbox_inches='tight')
plt.show()
