import utils
import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import os

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

confi = [1e-03, 3, 20, "tanh", "Glorot normal", 1, 1, 100]
a = utils.create_system(confi)
# ss = utils.train_model(a, "sys")
ss = utils.restore_model(a, "sys")
utils.plot_3d(ss)


# b = utils.create_observer(confi)
# oo = utils.train_model(b, "obs")
# # oo = utils.restore_model(b, "obs")
# utils.plot_obs(ss, oo)


W_tot = np.linspace(0.45, 4, num=8)
multi_obs = []
for j in range(len(W_tot)):
    model = utils.create_multi_obs(confi, W_tot[j])
    # modelu = utils.train_model(model, f"obs{j}")
    modelu = utils.restore_model(model, f"obs{j}")
    multi_obs.append(modelu)

e = np.linspace(0, 1, 10)

jj = [0, 1, 2, 3, 4, 5, 6, 7]

m = np.zeros((len(e), len(jj)))


for el in range(len(e)):
    m[el] = utils.mu(ss, multi_obs, el)

fig = plt.figure()
ax1 = fig.add_subplot(111)
for z in range(len(jj)):
    plt.plot(e, m[:, z], alpha=1.0, linewidth=.8, color=f'C{z}', label=f"$obs{jj[z]}$")


ax1.set_xlabel(xlabel=r"${\tau}$", fontsize=14.0)  # xlabel
ax1.set_ylabel(ylabel=r"$\mu(\tau)$", fontsize=14.0)  # ylabel
ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax1.set_title(r"$\mu(\tau)$")
plt.savefig(f"{folder_path}figures/mu.png",
           dpi=150, bbox_inches='tight')

plt.show()