import utils
import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import os

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

# confi = [0.014, 1, 75, "tanh", "Glorot normal", 1, 1, 1, 1]
# a = utils.create_system(confi)
# ss = utils.train_model(a, "sys")
# ss = utils.restore_model(a, "sys")


# utils.inizia_hpo()
# output = utils.hpo(confi)
output = [0.0002889, 4, 61, "silu", "He normal", 81, 100, 26, 19]
ff = utils.create_system(output)
# gg = utils.train_model(ff, "syst_hpo")
gg = utils.restore_model(ff, "syst_hpo")
# utils.plot_hpo(ss, gg)


# b = utils.create_observer(output)
# oo = utils.train_model(b, "obs")
# oo = utils.restore_model(b, "obs")
# utils.plot_obs(gg, oo)


W_tot = np.linspace(0.36, 0.72, num=8)
multi_obs = []
for j in range(len(W_tot)):
    model = utils.create_multi_obs(output, W_tot[j])
    # modelu = utils.train_model(model, f"obs{j}")
    modelu = utils.restore_model(model, f"obs{j}")
    multi_obs.append(model)

p0 = np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
lam = 10000

def f(t, p):
    a = utils.mu(gg, multi_obs, t)
    e = np.exp(-1*a)
    d = np.inner(p, e)
    f = []
    for el in range(len(p)):
        ccc = - lam * (1-(e[el]/d))*p[el]
        f.append(ccc)
    return np.array(f)
    

sol = integrate.solve_ivp(f, (0, 1), p0, t_eval=np.linspace(0, 1, 100))
x = sol.y
t = sol.t

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
plt.savefig(f"{folder_path}figures/p_lam_{lam}.png", dpi=150, bbox_inches='tight')

plt.show()




