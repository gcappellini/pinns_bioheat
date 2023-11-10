import utils2
import numpy as np
import deepxde as dde
import wandb
import time

utils2.inizia()
confi = [0.0001046, 3, 165, "sigmoid", "He normal", 1, 1, 1, 10000]
ded = utils2.create_system(confi)
# o = utils2.train_model(d, "sys")
oeo = utils2.restore_model(ded, "sys")


# # Implement observer 1D
learning_rate, num_dense_layers, num_dense_nodes, activation, initialization, w_domain, w_bc0, w_bc1, w_ic = [0.00004285, 5, 26, "sigmoid", "Glorot uniform", 58, 164, 97, 164]
config = [0.00004285, 5, 26, "sigmoid", "Glorot uniform", 58, 164, 97, 164]


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="obs-simulation",

    # track hyperparameters and run metadata
    config={
        "learning rate": learning_rate,
        "num_dense_layers": num_dense_layers,
        "num_dense_nodes": num_dense_nodes,
        "activation": activation,
        "initialization": initialization,
        "w_domain": w_domain,
        "w_bc0": w_bc0,
        "w_bc1": w_bc1,
        "w_ic": w_ic        
    }
)


# Print the hyper-parameters.
print("learning rate: {0:.1e}".format(learning_rate))
print("num_dense_layers:", num_dense_layers)
print("num_dense_nodes:", num_dense_nodes)
print("activation:", activation)
print("initialization:", initialization)
print("w_domain:", w_domain)
print("w_bc0:", w_bc0)
print("w_bc1:", w_bc1)
print("w_ic:", w_ic)
print()

start_time = time.time()

# Create the neural network with these hyper-parameters.
mm = utils2.create_observer(config)
nn = utils2.train_model(mm, f"obs_ft1")
# possibility to change where we save
error = utils2.l2_penalty(oeo, nn, 25)
# print(accuracy, 'accuracy is')

if np.isnan(error):
    error = 10 ** 5

end_time = time.time()
time_spent = end_time - start_time

# Store the configuration and error in a DataFrame
data = {
    "Learning Rate": learning_rate,
    "Num Dense Layers": num_dense_layers,
    "Num Dense Nodes": num_dense_nodes,
    "Activation": activation,
    "Initialization": initialization,
    "W_domain": w_domain,
    "W_bc0": w_bc0,
    "W_bc1": w_bc1,
    "W_ic": w_ic,
    "Error": error,
    "Time Spent": time_spent
}


wandb.log({"err": error})
wandb.finish()



n_guide = 25
utils2.plot_1obs(oeo, nn, n_guide)
utils2.plot_1obs_tf(oeo, nn, n_guide)
utils2.plot_1obs_l2(oeo, nn, n_guide)
