import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dde.config.set_random_seed(1)
learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = [0.001, 1, 30, "elu", "Glorot normal"]
start_flux, end_time, w_domain, w_bcl, w_bcr, w_ic = [-1, 1, 1, 1, 1, 1]
folder_path = ""
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

# Antenna parameters
beta = 1
p = 150/(1.75e-3)
cc = 16
X0 = 0.08

# Considero il caso flusso costante
def gen_testdata():
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_system.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, 0*t, t)).T
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


geom = dde.geometry.Rectangle([0, start_flux], [1, 0])
timedomain = dde.geometry.TimeDomain(0, end_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
bc_1 = dde.icbc.DirichletBC(geomtime, lambda x: x[:, 1:2], boundary_1)

ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

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
model.train(iterations=epochs)
# # model.compile("L-BFGS")
# # model.train()
#
# X = geomtime.random_points(100000)
# err = 1
# while err > eee:
#     f = model.predict(X, operator=pde)
#     err_eq = np.absolute(f)
#     err = np.mean(err_eq)
#     print("Mean residual: %.3e" % (err))
#
#     x_id = np.argmax(err_eq)
#     print("Adding new point:", X[x_id], "\n")
#     data.add_anchors(X[x_id])
#     early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
#     model.compile("adam", lr=1e-3)
#     model.train(iterations=epochs, disregard_previous_best=True, callbacks=[early_stopping])
#     # model.compile("L-BFGS")
#     # losshistory, train_state = model.train()
#
#     losshistory, train_state = model.train(iterations=epochs, disregard_previous_best=True, callbacks=[early_stopping])
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x_true, y_true = gen_testdata()
y_pred = model.predict(x_true)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((x_true, y_true, y_pred)))

print("Mean absolute error:", np.mean(np.abs(y_true, y_pred)))
print("Maximum absolute error:", np.max(np.abs(y_true, y_pred)))
print("Standard deviation:", np.std(np.abs(y_true, y_pred)))

ll = int(np.sqrt(len(y_true)))

x_grid = np.reshape(x_true[:, 0:1], (ll, ll))
t_grid = np.reshape(x_true[:, 2:], (ll, ll))
y_true_grid = np.reshape(y_true, (ll, ll))
y_pred_grid = np.reshape(y_pred, (ll, ll))

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

ax1.view_init(20, -120)
ax2.view_init(20, -120)
ax3.view_init(20, -120)

ax1.plot_surface(x_grid, t_grid, y_pred_grid, cmap='inferno', alpha=.8)
ax2.plot_surface(x_grid, t_grid, y_true_grid, cmap='inferno', alpha=.8)
ax3.plot_surface(x_grid, t_grid, np.abs(y_pred_grid-y_true_grid), cmap='inferno', alpha=.8)

plt.savefig("figures/source3d.png", dpi=300, bbox_inches='tight')
plt.show()



