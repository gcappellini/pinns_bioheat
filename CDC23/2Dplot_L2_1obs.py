import numpy as np
from system import fun, y2
from observer import fun_1obs
import deepxde as dde
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)

"""Plotting l2 error over time"""
l2_k0 = []
for te in t:
    tau = np.full_like(x, te)
    y2_v = np.full_like(x, y2(te))

    Xs = np.vstack((x, tau)).T
    Xo = np.vstack((x, y2_v, tau)).T

    s = fun(Xs)
    o = fun_1obs(Xo)

    l2_k0.append(dde.metrics.l2_relative_error(s, o))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(1800*t, l2_k0, alpha=1.0, linewidth=1.8, color='C0')

ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
ax1.set_ylim(0, max(l2_k0))
ax1.set_xlim(0, 1801)
plt.yticks(np.arange(0, max(l2_k0), 0.2), fontsize=7)
plt.xticks(np.arange(0, 1801, 600), fontsize=7)

plt.grid()
ax1.set_box_aspect(1)
plt.savefig("figures/L2_1observer.png",
           dpi=300, bbox_inches='tight')
plt.show()




