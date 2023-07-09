import numpy as np
from system import fun, y2
from multi_observer import fun_obs
import deepxde as dde
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
a = np.load('new_weights_lambda_200.npy', allow_pickle=True)

TM = 45
Ta = 37

"""Plotting l2 error over time"""
l2_k0 = []
for te in range(len(t)):

    tau = np.full_like(x, t[te])
    y2_v = np.full_like(x, y2(t[te]))

    Xs = np.vstack((x, tau)).T
    Xo = np.vstack((x, y2_v, tau)).T

    s = Ta + fun(Xs)*(TM-Ta)
    o0f = Ta + fun_obs(0, Xo) * (TM - Ta)
    o1f = Ta + fun_obs(1, Xo) * (TM - Ta)
    o2f = Ta + fun_obs(2, Xo) * (TM - Ta)
    o3f = Ta + fun_obs(3, Xo) * (TM - Ta)
    o4f = Ta + fun_obs(4, Xo) * (TM - Ta)
    o5f = Ta + fun_obs(5, Xo) * (TM - Ta)
    o6f = Ta + fun_obs(6, Xo) * (TM - Ta)
    o7f = Ta + fun_obs(7, Xo) * (TM - Ta)
    o = a[1, te] * o0f + a[2, te] * o1f + a[3, te] * o2f + a[4, te] * o3f + a[5, te] * o4f + a[6, te] * o5f + \
        a[7, te] * o6f + a[8, te] * o7f

    l2_k0.append(dde.metrics.l2_relative_error(s, o))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(1800*t, l2_k0, alpha=1.0, linewidth=1.8, color='C0')

ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
ax1.set_ylim(bottom=0.0)
ax1.set_xlim(0, 1801)
plt.yticks(fontsize=7)
plt.xticks(np.arange(0, 1801, 600), fontsize=7)

plt.grid()
ax1.set_box_aspect(1)
plt.savefig("figures/L2_8observers.png",
           dpi=300, bbox_inches='tight')
plt.show()




