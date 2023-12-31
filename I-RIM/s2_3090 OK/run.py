import utils
import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import os

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_tot = np.linspace(0.45, 4, num=8)
W_ref = W_tot[4]
confi = [0.0008729, 4, 37, "silu", "He normal", 28, 99, 37, 37]
# utils.inizia_hpo()
# # a = utils.hpo(confi)
# a = [0.0008729, 4, 37, "silu", "He normal", 28, 99, 37, 37]
b = utils.create_system(confi, W_ref)
ss = utils.train_model(b, "sys")
# ss = utils.restore_model(b, "sys")
utils.sup_theta(ss)
utils.plot_3d(ss)
utils.plot_3d_sys(ss)

# confi2 = [0.06552, 1, 189, "selu", "He normal", 93, 51, 43, 158]
# utils.inizia_hpo()
# c = utils.hpo(confi2)
c = [0.0001046, 4, 265, "sigmoid", "He normal", 109, 125, 156, 178]
d = utils.create_observer(c, W_ref)
oo = utils.train_model(d, "obs")
# print(oo)
# print("----------------")
# oo = utils.restore_model(d, "obs")
# utils.plot_3d_obs(oo)

utils.plot_1obs(ss, oo)
utils.plot_1obs_tf(ss, oo)
utils.plot_1obs_l2(ss, oo)



multi_obs = []
for j in range(len(W_tot)):
    model = utils.create_observer(confi, W_tot[j])
    modelu = utils.train_model(model, f"obs{j}")
    # modelu = utils.restore_model(model, f"obs{j}")
    multi_obs.append(modelu)

p0 = np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
lem = [5, 20, 200]

for lam in lem:
# lam = 5
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
    weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    np.save(f'{folder_path}pinns/weights_lambda_{lam}.npy', weights)
    utils.plot_weights(x, t, lam)

utils.plot_8obs_tf(ss, multi_obs)
utils.plot_8obs_l2(ss, multi_obs)






