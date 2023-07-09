import numpy as np
from system import fun, y2
from observer import fun_1obs
import deepxde as dde
import matplotlib.pyplot as plt

TM = 45
Ta = 37

x = np.linspace(0, 1, 51)
t = np.linspace(0, 1, 51)

"""Plotting system and observer at tau=1"""

tauf = np.ones_like(x)
y2_v = np.full_like(x, y2(1))
xs = np.vstack((x, tauf)).T
xo = np.vstack((x, y2_v, tauf)).T

sf = Ta + fun(xs)*(TM - Ta)
of = Ta + fun_1obs(xo)*(TM - Ta)

fig = plt.figure()

ax2 = fig.add_subplot(111)
ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', color='C0', label="System")
ax2.plot(x, of, linestyle='None', marker="x", color='C1', label="Observer")

ax2.set_ylim(36, 47)
ax2.set_xlim(0, 1)
plt.yticks(np.arange(36, 47.01, 2))
plt.xticks(np.arange(0, 1.01, 0.1))
ax2.legend()
ax2.set_ylabel(ylabel=r"Temperature")
ax2.set_xlabel(xlabel=r"Distance $x$")
ax2.set_title(r"Solution at $t=t_f$", weight='semibold')

plt.grid()

plt.savefig("figures/tf_1observer.png",
           dpi=150, bbox_inches='tight')
plt.show()




