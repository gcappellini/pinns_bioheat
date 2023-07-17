import utils
import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import os

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

confi = [0.014, 1, 75, "tanh", "Glorot normal", 1, 1, 1, 1]
# a = utils.create_system(confi)
# ss = utils.train_model(a, "sys")
# ss = utils.restore_model(a, "sys")
# utils.plot_3d(ss)


# utils.inizia_hpo()
# output = utils.hpo(confi)
output = [0.0002889, 4, 61, "silu", "He normal", 19, 100, 26, 81]
ff = utils.create_system(output)
# gg = utils.train_model(ff, "syst_hpo")
gg = utils.restore_model(ff, "syst_hpo")
# utils.plot_hpo(ss, gg)

# utils.inizia_hpo()
# output2 = utils.hpo(output)
output2 = [0.0019886835004823537, 4, 176, 'sin', 'He uniform', 54, 22, 26, 9]
b = utils.create_observer(confi)
# oo = utils.train_model(b, "obs_hpo")
oo = utils.restore_model(b, "obs_hpo")
# utils.plot_obs(gg, oo)


W_tot = np.linspace(0.45, 4, num=8)
multi_obs = []
for j in range(len(W_tot)):
    model = utils.create_multi_obs(confi, W_tot[j])
    # modelu = utils.train_model(model, f"obs{j}_hpo")
    modelu = utils.restore_model(model, f"obs{j}_hpo")
    multi_obs.append(modelu)

p0 = np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
lam = 2000

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
# weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
# weights[0] = sol.t
# weights[1:] = sol.y
# np.save(f'{folder_path}weights_lambda_{lam}.npy', weights)
utils.plot_weights(x, t, lam)






