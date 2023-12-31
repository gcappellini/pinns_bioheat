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
    utils.plot_obs(ss, modelu, f"obs{j}")

p0 = np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
lam = 10000

def f(t, p):
    a = utils.mu(ss, multi_obs, t)
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
# weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
# weights[0] = sol.t
# weights[1:] = sol.y
# np.save(f'{folder_path}weights_lambda_{lam}.npy', weights)
utils.plot_weights(x, t, lam)






