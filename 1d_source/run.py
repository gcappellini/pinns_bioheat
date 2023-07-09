import utils

confi = [0.014, 1, 75, "tanh", "Glorot normal", 1, 1, 1, 1, 1]
a = utils.create_system(confi)
b = utils.create_observer(confi)

ss = utils.restore_model(a, "sys")
oo = utils.restore_model(b, "obs")
utils.plot_obs(ss, oo)




