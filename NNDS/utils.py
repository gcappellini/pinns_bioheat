import deepxde as dde
import os
import numpy as np
import torch
import tensorflow as tf

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

epochs = 20000
dde_seed = 200

# General parameters
L0, TM, Ta, tauf, qmet = 0.05, 45, 37, 1800, 4200

# Tissue parameters
# rho, c, k_eff, W_min, W_avg, W_max = 888, 2387, 1.2, 0.36, 0.54, 0.72           # fat
rho, c, k_eff, W_min, W_avg, W_max = 1050, 3639, 5, 0.45, 2.3, 4           # muscle
cb = 3825

dT = TM - Ta
alfa = rho * c / k_eff

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)

# Antenna parameters
beta, cc, X0, p = 1, 16, 0.09, 150/(1.97e-3)


class SReLU(tf.keras.layers.Layer):
    def __init__(self, tr_init=1.0, ar_init=0.1, tl_init=-1.0, al_init=0.1):
        super(SReLU, self).__init__()
        self.tr = tf.Variable(tr_init, trainable=True, name='tr')
        self.ar = tf.Variable(ar_init, trainable=True, name='ar')
        self.tl = tf.Variable(tl_init, trainable=True, name='tl')
        self.al = tf.Variable(al_init, trainable=True, name='al')

    def call(self, inputs):
        s_greater_tr = tf.where(inputs > self.tr, self.tr + self.ar * (inputs - self.tr), inputs)
        s_between_tl_tr = tf.where(tf.logical_and(inputs > self.tl, inputs <= self.tr), inputs, s_greater_tr)
        s_less_tl = tf.where(inputs < self.tl, self.tl + self.al * (inputs - self.tl), s_between_tl_tr)
        return s_less_tl


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def source(s):
    return qmet + beta*torch.exp(-cc*L0*(X0-s))*p


def pde_s(x, theta, W):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)
    return a1 * dtheta_tau - dtheta_xx + a2 * W * theta - a3 * source(x[:, 0:1])


def create_system(config, W):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bcl, w_bcr, w_ic = config

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: -1, boundary_1)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, lambda x, theta: pde_s(x, theta, W), [bc_0, bc_1, ic], num_domain=2560, num_boundary=100,
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
    global dde_seed, output_path
    dde.config.set_random_seed(dde_seed)
    # early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)

    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{folder_path}model/{name}.ckpt")#,
                                        #    callbacks=[early_stopping])
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{name}_loss",
                 train_fname=f"{name}_train", test_fname=f"{name}_test",
                 output_dir=f"{folder_path}history")
    return model


def restore_model(model, name):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    model.restore(f"{folder_path}model/{name}.ckpt-{epochs}.pt", verbose=0)
    return model


def gen_sysdata():
    data = np.loadtxt(f"{folder_path}matlab/output_matlab_pde.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y

# a = [0.0008729, 3, 25, "silu", "He normal", 1, 1, 1, 1]
# W_tot = np.linspace(W_min, W_max, num=8)
# W_ref = W_tot[3]
#
# b = create_system(a, W_ref)
# ss = train_model(b, "sys")
# # ss = restore_model(b, "sys")
#
# XO, y_true = gen_sysdata()
# y_pred = ss.predict(XO)
# er = dde.metrics.l2_relative_error(y_true, y_pred)
#
# print(f"error is: {er}")
