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

W_tot = np.linspace(0.45, 4, num=8)
W_ref = W_tot[3]
W_min = 0.45
W_max = 4
cb = 3825

dT = TM - Ta

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)

#Observer adaptive gain
k = 4


def source(s):
    return qmet


def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W_ref * theta - a3 * source(x[:, 0:1])

def pde_m(x, theta, W):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=3)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W * theta - a3 * source(x[:, 0:1])


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def create_system(config):

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: -1, boundary_1)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde(x, theta), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100,
        num_initial=160
    )

    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    model = dde.Model(data, net)

    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def train_model(model, name):
    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{folder_path}model/{name}.ckpt")#,
                                        #    callbacks=[early_stopping])
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{name}_loss",
                 train_fname=f"{name}_train", test_fname=f"{name}_test",
                 output_dir=f"{folder_path}history")
    return model


def restore_model(model, name):
    model.restore(f"{folder_path}model/{name}.ckpt-{epochs}.pt", verbose=0)
    return model


def sup_theta(e):
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    xsup = np.vstack((np.ones_like(np.ravel(T)), np.ravel(T))).T
    xx = np.vstack((np.ravel(X), np.ravel(T))).T
    a = e.predict(xsup).T
    b = e.predict(xx).T
    np.savez(f"{folder_path}pinns/sup_theta.npz", data=a)
    np.savez(f"{folder_path}pinns/theta.npz", data=b)


def create_observer(config, W):
    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    xmin = [0, 0, -2]
    xmax = [1, 1, 0]
    geom = dde.geometry.Cuboid(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_m(x, theta, W), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100,
        num_initial=160
    )

    net = dde.nn.FNN(
        [4] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )

    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]

    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def ic_obs(x):
    return x[:, 0:1] * (6/5 - x[:, 0:1])



def bc1_obs(x, theta, X):
    dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    return dtheta_x - x[:, 2:3] - k * (x[:, 1:2] - theta)




def plot_1obs_tf(ss, oo):

    x = np.linspace(0, 1, 51)
    e = np.vstack((np.ones_like(x), np.ones_like(x))).T
    sup_theta = ss.predict(e).reshape(x.shape)

    fl = np.full_like(x, -1)
    t = np.ones_like(x)
    XS = np.vstack((x, t)).T
    XO = np.vstack((x, sup_theta, fl, t)).T

    sf = ss.predict(XS)
    of = oo.predict(XO)

    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', color='C0', label="System")
    ax2.plot(x, of, linestyle='None', marker="x", color='C1', label="Observer")

    ax2.legend()
    ax2.set_ylabel(ylabel=r"Temperature")
    ax2.set_xlabel(xlabel=r"Distance $x$")
    ax2.set_title(r"Solution at $t=t_f$", weight='semibold')

    plt.grid()

    plt.savefig(f"{folder_path}figures/tf_1observer.png",
                dpi=150, bbox_inches='tight')
    plt.show()


def mu(s, o, tau):
    xs = np.vstack((np.ones_like(tau), tau)).T
    th = s.predict(xs)
    fl = np.full_like(tau, -1)
    xo = np.vstack((np.ones_like(tau), th, fl, tau)).T
    muu = []
    for el in o:
        oss = el.predict(xo)
        scrt = np.abs(oss-th)
        muu.append(scrt)
    muu = np.array(muu).reshape(len(muu),)
    return muu


def plot_weights(x, t, lam):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ['C3', 'lime', 'blue', 'purple', 'aqua', 'lightskyblue', 'darkred', 'k']

    for i in range(x.shape[0]):
        plt.plot(1800 * t, x[i], alpha=1.0, linewidth=1.8, color=colors[i], label=f"Weight $p_{i+1}$")

    ax1.set_xlim(0, 1800)
    ax1.set_ylim(bottom=0.0)

    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(f"Dynamic weights, $\lambda={lam}$", weight='semibold')
    plt.grid()
    plt.savefig(f"{folder_path}figures/weights_lam_{lam}.png", dpi=150, bbox_inches='tight')

    plt.show()


def plot_8obs_tf(ss, multi_obs):
    a = np.load(f'{folder_path}pinns/weights_lambda_200.npy', allow_pickle=True)
    y = a[1:]
    w_f = y[:, -1]
    x = np.linspace(0, 1, 101)
    e = np.vstack((np.ones_like(x), np.ones_like(x))).T
    sup_theta = ss.predict(e).reshape(x.shape)

    fl = np.full_like(x, -1)
    t = np.ones_like(x)
    XS = np.vstack((x, t)).T
    XO = np.vstack((x, sup_theta, fl, t)).T

    sf = ss.predict(XS)
    o0f = multi_obs[0].predict(XO)
    o1f = multi_obs[1].predict(XO)
    o2f = multi_obs[2].predict(XO)
    o3f = multi_obs[3].predict(XO)
    o4f = multi_obs[4].predict(XO)
    o5f = multi_obs[5].predict(XO)
    o6f = multi_obs[6].predict(XO)
    o7f = multi_obs[7].predict(XO)
    o8f = w_f[0] * o0f + w_f[1] * o1f + w_f[2] * o2f + w_f[3] * o3f + w_f[4] * o4f + w_f[5] * o5f + w_f[6] * o6f + w_f[
        7] * o7f

    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', color='C0', label="System")
    ax2.plot(x, o0f, alpha=1.0, linewidth=1.8, color='C3', label="Observer1")
    ax2.plot(x, o1f, alpha=1.0, linewidth=1.8, color='lime', label="Observer2")
    ax2.plot(x, o2f, alpha=1.0, linewidth=1.8, color='blue', label="Observer3")
    ax2.plot(x, o3f, alpha=1.0, linewidth=1.8, color='aqua', label="Observer4")
    ax2.plot(x, o4f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='lightskyblue', label="Observer5")
    ax2.plot(x, o5f, alpha=1.0, linestyle="dashdot", linewidth=1.8, color='darkred', label="Observer6")
    ax2.plot(x, o6f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='deeppink', label="Observer7")
    ax2.plot(x, o7f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='purple', label="Observer8")
    ax2.plot(x, o8f, linestyle='None', marker="X", color='gold', label="MM adaptive observer")

    ax2.set_xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 0.1))
    ax2.legend()
    ax2.set_ylabel(ylabel=r"Temperature")
    ax2.set_xlabel(xlabel=r"Distance $x$")
    ax2.set_title(r"Solutions at $t=t_f$", weight='semibold')

    plt.grid()

    plt.savefig(f"{folder_path}figures/tf_8observers.png",
                dpi=150, bbox_inches='tight')
    plt.show()

