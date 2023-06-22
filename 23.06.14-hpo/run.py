import utils


for _ in range(10):
    confi = [0.014, 1, 75, "tanh"] 
    utils.esegui_esperimento()
    output = utils.hpo(confi)
    utils.plot_3d(output)
    print("----------------------------------")
    utils.reset_iteration()

utils.data_analysis("/home/giuglielmocappellini/Projects/PINNs/23.06.14_hpo2/output")