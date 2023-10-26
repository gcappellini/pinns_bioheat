"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

dde.config.set_random_seed(112)

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


def source(s):
    return qmet + beta*torch.exp(-cc*L0*(X0-s))*p


def gen_traindata(num=None):
    data = np.loadtxt(f"{folder_path}output_matlab_pde.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]

    # If num is not specified, return the entire X and y
    if num is None:
        return X, y

    indices = np.random.choice(X.shape[0], size=num, replace=False)

    # Extract the samples
    X_sample = X[indices]
    y_sample = y[indices]
    return X_sample, y_sample


def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W_avg * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ob_x, ob_u = gen_traindata()
observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)

data = dde.data.PDE(
    geomtime,
    pde,
    [observe_u],
    num_domain=200,
    anchors=ob_x,
    num_test=1000,
)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1],
    activation,
    initialization,
)

loss_weights = [1, 1e+05]
model = dde.Model(data, net)
model.compile("adam", lr=0.0001, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=20000, model_save_path=f"{folder_path}model/system.ckpt")

# model.compile("L-BFGS")
# model.restore(f"{folder_path}model/system.ckpt-20000.pt", verbose=0)


X, y_true = gen_traindata()
y_pred = model.predict(X)

mean_residual = np.mean(np.absolute(y_pred))
l2_error = dde.metrics.l2_relative_error(y_true, y_pred)

print("Mean residual:", mean_residual)
print("L2 relative error:", l2_error)


ll = int(np.sqrt(len(y_true)))

XC = X[:, 0:1].reshape(ll, ll)
T = X[:, 1:].reshape(ll, ll)
Y_pred = y_pred.reshape(ll, ll)
Y_true = y_true.reshape(ll, ll)
Y = np.abs(Y_true - Y_pred)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

# Plot Y_true
axs[0].plot_surface(XC, T, Y_true, cmap='inferno', alpha=.8)
axs[0].set_title("Matlab")
axs[0].set_ylabel("X")
axs[0].set_xlabel("Time")
axs[0].set_zlabel("Temperature")
axs[0].view_init(20, 20)

# Plot Y_pred
axs[1].plot_surface(XC, T, Y_pred, cmap='inferno', alpha=.8)
axs[1].set_title("PINNs")
axs[1].set_ylabel("X")
axs[1].set_xlabel("Time")
axs[1].set_zlabel("Temperature")
axs[1].view_init(20, 20)

# Plot Y
axs[2].plot_surface(XC, T, Y, cmap='inferno', alpha=.8)
axs[2].set_title("Error")
axs[2].set_ylabel("X")
axs[2].set_xlabel("Time")
axs[2].set_zlabel("Temperature")
axs[2].view_init(20, 20)

# Add text information
fig.text(0.5, 0.01, f"Mean residual: {mean_residual:.4f}", fontsize=12)
fig.text(0.1, 0.01, f"L2 relative error: {l2_error:.4f}", fontsize=12)

plt.tight_layout()
plt.savefig(f"{folder_path}figures/system_sample_weight_e5.png", dpi=300, bbox_inches='tight')
plt.show()
