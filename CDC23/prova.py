from __future__ import print_function
import numpy as np
import torch
from multi_observer import fun_obs
from system import y2
import matplotlib.pyplot as plt


def mu(j, tau):
    ys = y2(tau)
    xo = np.vstack((np.ones_like(tau),
                    y2(tau),
                    tau)).T
    yo = fun_obs(j, xo)
    a = torch.abs(ys-yo)
    return a


e = np.linspace(0, 1, 10)
jj = [0, 1, 2, 3, 4, 5, 6, 7]

m = np.zeros((len(e), len(jj)))

for jl in range(len(jj)):
    for el in range(len(e)):
        m[el, jl] = mu(jj[jl], e[el])


fig = plt.figure()
ax1 = fig.add_subplot(111)


for z in range(len(jj)):
    plt.plot(e, m[:, z], alpha=1.0, linewidth=.8, color='C{}'.format(z), label="$obs{}$".format(jj[z]))

ax1.set_xlabel(xlabel=r"${\tau}$", fontsize=14.0)  # xlabel
ax1.set_ylabel(ylabel=r"$\mu(\tau)$", fontsize=14.0)  # ylabel
ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax1.set_title(r"$\mu(\tau)$")
plt.savefig("figures/mu.png",
           dpi=150, bbox_inches='tight')

plt.show()
