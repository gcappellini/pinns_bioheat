import utils2
import numpy as np
import deepxde as dde

utils2.inizia()
confi = [0.0001046, 3, 165, "sigmoid", "He normal", 1, 1, 1, 10000]
ded = utils2.create_system(confi)
# o = utils2.train_model(d, "sys")
oeo = utils2.restore_model(ded, "sys")

time_instants = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]
# Iterate over the time instants and plot the predictions for each
for t in time_instants:
    utils2.plot(oeo, t)
utils2.plot_needle(oeo)



# # Implement observer 1D
e = [0.00004285, 5, 26, "sigmoid", "Glorot uniform", 58, 164, 97, 164]
n = utils2.create_observer(e)
m = utils2.train_model(n, "obs")
# # m = utils2.restore_model(n, "obs")

# utils2.inizia_hpo()
# a = utils2.hpo(e)

n_guide = 25
utils2.plot_1obs(oeo, m, n_guide)
utils2.plot_1obs_tf(oeo, m, n_guide)
utils2.plot_1obs_l2(oeo, m, n_guide)




