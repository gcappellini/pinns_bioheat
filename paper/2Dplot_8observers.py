import numpy as np
from system import fun, y2
from multi_observer import fun_obs
import deepxde as dde
import matplotlib.pyplot as plt

TM = 45
Ta = 37
a = np.load('new_weights_lambda_500.npy', allow_pickle=True)
y = a[1:]
w_f = y[:,-1]
# x = np.linspace(0.665, 0.705, 51)
x = np.linspace(0, 1, 51)
t = np.linspace(0, 1, 51)

"""Plotting system and observer at tau=1"""

tauf = np.ones_like(x)
y2_v = np.full_like(x, y2(1))
xs = np.vstack((x, tauf)).T
xo = np.vstack((x, y2_v, tauf)).T

sf = Ta + fun(xs)*(TM - Ta)
o0f = Ta + fun_obs(0, xo)*(TM - Ta)
o1f = Ta + fun_obs(1, xo)*(TM - Ta)
o2f = Ta + fun_obs(2, xo)*(TM - Ta)
o3f = Ta + fun_obs(3, xo)*(TM - Ta)
o4f = Ta + fun_obs(4, xo)*(TM - Ta)
o5f = Ta + fun_obs(5, xo)*(TM - Ta)
o6f = Ta + fun_obs(6, xo)*(TM - Ta)
o7f = Ta + fun_obs(7, xo)*(TM - Ta)
o8f = w_f[0]*o0f + w_f[1]*o1f + w_f[2]*o2f + w_f[3]*o3f + w_f[4]*o4f + w_f[5]*o5f + w_f[6]*o6f + w_f[7]*o7f

fig = plt.figure()

ax2 = fig.add_subplot(111)
ax2.plot(x, sf, linestyle='None', marker="o", mfc='none', color='C0', label="System")
ax2.plot(x, o0f, alpha=1.0, linewidth=1.8, color='C3', label="Observer1")
ax2.plot(x, o1f, alpha=1.0, linewidth=1.8, color='lime', label="Observer2")
ax2.plot(x, o2f, alpha=1.0, linewidth=1.8, color='blue', label="Observer3")
ax2.plot(x, o3f, alpha=1.0, linewidth=1.8, color='aqua', label="Observer4")
ax2.plot(x, o4f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='lightskyblue', label="Observer5")
ax2.plot(x, o5f, alpha=1.0, linestyle="dashdot",linewidth=1.8, color='darkred', label="Observer6")
ax2.plot(x, o6f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='deeppink', label="Observer7")
ax2.plot(x, o7f, alpha=1.0, linestyle="dashed", linewidth=1.8, color='purple', label="Observer8")
ax2.plot(x, o8f, linestyle='None', marker="X", color='gold', label="MM adaptive observer")


# ax2.set_ylim(39.5, 42)
ax2.set_ylim(36, 47)
# ax2.set_xlim(0.665, 0.705)
ax2.set_xlim(0, 1)
# plt.yticks(np.arange(39.5, 42, 0.2))
plt.yticks(np.arange(36, 47.01, 2))
# plt.xticks(np.arange(0.665, 0.706, 0.005))
plt.xticks(np.arange(0, 1.01, 0.1))
ax2.legend()
ax2.set_ylabel(ylabel=r"Temperature")
ax2.set_xlabel(xlabel=r"Distance $x$")
ax2.set_title(r"Solutions at $t=t_f$", weight='semibold')

plt.grid()

# plt.savefig("figures/tf_8observers_zoomed.png",
#            dpi=150, bbox_inches='tight')
plt.savefig("figures/tf_8observers.png",
           dpi=150, bbox_inches='tight')
plt.show()




