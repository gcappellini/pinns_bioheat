import utils
import os

for _ in range(9):
    confi = [1.4e-2, 1, 75, "tanh"]
    utils.esegui_esperimento()
    output = utils.hpo(confi)
    utils.plot_3d(output)
    print("----------------------------------")
    utils.reset_iteration()











