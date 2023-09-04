import utils
import numpy as np
import os
import matplotlib.pyplot as plt
import deepxde as dde

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

W_tot = np.linspace(0.45, 4, num=8)
W_ref = W_tot[3]
a = [0.0008729, 4, 37, "silu", "He normal", 28, 99, 37, 37]
b = utils.create_system(a, W_ref)
ss = utils.restore_model(b, "sys")
c = [0.0001046, 4, 265, "sigmoid", "He normal", 109, 125, 156, 178]
d = utils.create_observer(c, W_ref)
oo = utils.restore_model(d, "obs")

multi_obs = []
for j in range(len(W_tot)):
    model = utils.create_observer(c, W_tot[j])
    # modelu = utils.train_model(model, f"obs{j}")
    modelu = utils.restore_model(model, f"obs{j}")
    multi_obs.append(modelu)

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
fl = np.full_like(x, -1)
a = np.load(f'{folder_path}pinns/weights_lambda_200.npy', allow_pickle=True)

"""Plotting l2 error over time"""
l2_k0 = []
for te in range(len(t)):

    tau = np.full_like(x, t[te])
    e = np.vstack((np.ones_like(x), tau)).T
    sup_theta = ss.predict(e).reshape(x.shape)

    XS = np.vstack((x, tau)).T
    XO = np.vstack((x, sup_theta, fl, tau)).T

    sf = ss.predict(XS)
    o0f = multi_obs[0].predict(XO)
    o1f = multi_obs[1].predict(XO)
    o2f = multi_obs[2].predict(XO)
    o3f = multi_obs[3].predict(XO)
    o4f = multi_obs[4].predict(XO)
    o5f = multi_obs[5].predict(XO)
    o6f = multi_obs[6].predict(XO)
    o7f = multi_obs[7].predict(XO)
    o = a[1, te] * o0f + a[2, te] * o1f + a[3, te] * o2f + a[4, te] * o3f + a[5, te] * o4f + a[6, te] * o5f + \
        a[7, te] * o6f + a[8, te] * o7f

    l2_k0.append(dde.metrics.l2_relative_error(sf, o))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(t, l2_k0, alpha=1.0, linewidth=1.8, color='C0')

ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
ax1.set_ylim(bottom=0.0)
ax1.set_xlim(0, 1.01)
plt.yticks(fontsize=7)

plt.grid()
ax1.set_box_aspect(1)
plt.savefig("figures/L2_8observers.png",
           dpi=300, bbox_inches='tight')
plt.show()


