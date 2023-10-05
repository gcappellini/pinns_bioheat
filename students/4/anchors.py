"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = [0.001, 1, 30, "elu", "Glorot normal"]

# General parameters
L0, TM, Ta, tauf, qmet = 0.05, 45, 37, 1800, 4200

# Tissue parameters
rho, c, k_eff, W_min, W_avg, W_max, cb = 888, 2387, 1.2, 0.36, 0.54, 0.72, 3825           # fat
# rho, c, k_eff, W_min, W_avg, W_max, cb = 1050, 3639, 5, 0.45, 2.3, 4, 3825           # muscle

dT = TM - Ta
alfa = rho * c / k_eff

a1, a2, a3 = (alfa * (L0 ** 2)) / tauf, (L0 ** 2) * cb / k_eff, (L0 ** 2) / (k_eff * dT)


def gen_traindata(num=None):
    data = np.loadtxt(f"output_matlab_pde.txt")
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
    return a1 * dtheta_tau - dtheta_xx + a2 * 2.3 * theta


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ob_x, ob_u = gen_traindata(100)
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

model = dde.Model(data, net)
model.compile("adam", lr=0.0001)
losshistory, train_state = model.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


x, ytrue = gen_traindata()
yhat = model.predict(x)


print("l2 relative error: " + str(dde.metrics.l2_relative_error(ytrue, yhat)))
plt.figure()
plt.plot(x, ytrue, "-", label="y_true")
plt.plot(x, yhat, "--", label="u_NN")
plt.legend()


plt.show()
