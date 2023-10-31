import utils2
import numpy as np

utils2.inizia()
c = [0.0001046, 2, 65, "sigmoid", "He normal", 1, 1, 1, 100]
d = utils2.create_system(c)
o = utils2.train_model(d, "sys")
# o = utils2.restore_model(d, "sys")

# Define the number of time instants
num_time_instants = 11

# Create an array of time instants from 0 to 1
time_instants = np.linspace(0, 1, num_time_instants)

# Iterate over the time instants and plot the predictions for each
for t in time_instants:
    utils2.plot(o, t)
