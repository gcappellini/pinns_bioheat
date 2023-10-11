"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

# if dde.backend.backend_name == "pytorch":
#     sin = dde.backend.pytorch.sin
# elif dde.backend.backend_name == "paddle":
#     sin = dde.backend.paddle.sin
# else:
#     from deepxde.backend import tf
#
#     sin = tf.sin


current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

# Problem parameters:
a = 0.4  # Thermal diffusivity
L = 1  # Length of the bar
n = 1  # Frequency of the sinusoidal initial conditions
k = 1


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Load the data:
    # data = np.load("theta.npz")
    # x, theta = data["X"], data["theta"].T

    data = np.load(f"{folder_path}sup_theta.npz")
    x_int, ysup, fl, theta = data["x_int"], data["ysup"], data["fl"], data["theta"]


    X = np.hstack((x_int[:, 0:1], ysup, fl, x_int[:, 1:]))
    y = theta.flatten()[:, None]
    return X, y


def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    # return dtheta_x + x[:, 2:3] + k * theta - k * x[:, 1:2]
    return dtheta_x - x[:, 2:3] - k * (x[:, 1:2] - theta)

def pde(x, y):
    """Expresses the PDE residual of the heat equation."""
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx

X, y_true = gen_testdata()

def ic_obs(x):
    a2 = 3
    y2 = X[0, 1:2]
    y3 = X[0, 2:3]
    return x[:, 0:1] * (k*y2 + y3 - a2*(k+2))/(k+1) + a2 * x[:, 0:1]**2

# Computational geometry:
geom = dde.geometry.Cuboid([0, 0, 0], [L, 1, 3])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)
ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_0, bc_1, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)
net = dde.nn.FNN([4] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Build and train the model:
# model.compile("adam", lr=1e-3)
# model.train(iterations=20000)
# model.compile("L-BFGS")
# losshistory, train_state = model.train(model_save_path="observer.ckpt")

model.compile("L-BFGS")
model.restore(f"{folder_path}model/observer.ckpt-35000.pt", verbose=0)

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
Y = np.abs(Y_true-Y_pred)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel("X")
ax1.set_ylabel("tempo")
ax1.set_zlabel("Temperature")

ax1.view_init(20, 20)

ax1.plot_surface(XC, T, Y_pred, cmap='inferno', alpha=.8)
plt.savefig(f"{folder_path}figures/obs.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
