import deepxde as dde
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dde.config.set_random_seed(1)
learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = [0.001, 1, 30, "elu", "Glorot normal"]
end_flux, end_time, w_domain, w_bcl, w_bcr, w_ic = [1, 1, 1, 1, 1, 1]
folder_path = ""
epochs = 20000
eee = 0.005

# General parameters
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

# Considero il caso flusso costante
def gen_testdata():
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_system_0.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    # xx, tt = np.meshgrid(x, t)
    X = np.vstack((x, np.full_like(x, q0_ad), t)).T
    y = exact.flatten()[:, None]
    return X, y

def pde(x, theta):

    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=2)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

    return a1 * dtheta_tau - dtheta_xx + a2 * theta * W_avg


def ic_func(x):
    return x[:, 1:2] * (x[:, 0:1] ** 4) / 4 + 15 * (((x[:, 0:1] - 1) ** 2) * x[:, 0:1]) / dT


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


geom = dde.geometry.Rectangle([0, 0], [1, end_flux])
timedomain = dde.geometry.TimeDomain(0, end_time)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: x[:, 1:2], boundary_1)

ic = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

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
# model.compile("L-BFGS")
# model.train()

X = geomtime.random_points(100000)
err = 1
while err > eee:
    f = model.predict(X, operator=pde)
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    x_id = np.argmax(err_eq)
    print("Adding new point:", X[x_id], "\n")
    data.add_anchors(X[x_id])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model.compile("adam", lr=1e-3)
    # model.train(iterations=epochs, disregard_previous_best=True, callbacks=[early_stopping])
    # model.compile("L-BFGS")
    # losshistory, train_state = model.train()

    losshistory, train_state = model.train(iterations=epochs, disregard_previous_best=True, callbacks=[early_stopping])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X_true, y_true = gen_testdata()
y_pred = model.predict(X_true)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


