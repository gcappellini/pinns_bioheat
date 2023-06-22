import utils

# scegli la configurazione
# confi = [0.014, 1, 75, 'tanh']
confi = [1.79e-03, 2, 500, "elu"]
# scegli i settings
# setts = [1.3, 10000, 5, 1]
setts = [10, 10000, 2, 0 ]
# scegli i seed e l'inizializzazione
initial = [46, "He normal"]

utils.inizia_esperimento(initial, confi, setts)
a = utils.create_model(initial, confi, setts)
b = utils.train_model(a, initial, confi, setts)

utils.plot_3d(initial, confi, setts)
utils.plot_err(initial, confi, setts)