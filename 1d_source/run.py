import utils
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import os
import torch
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dde.config.set_random_seed(1)

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

cc = [0.001, 1, 80, "elu", "Glorot normal", -1, 1, 1, 1, 1, 1, 1]
a = utils.create_model(cc)
b = utils.restore_model(a)
# b = utils.train_model(a)

x = np.linspace(0,1, num=101)
t = np.linspace(0,1, num=101)

xx, tt = np.meshgrid(x, t)

fl = -0.8*np.ravel(tt)

grid = np.vstack((np.ravel(xx), fl, np.ravel(tt))).T
e = b.predict(grid)
e_grid = np.reshape(e, xx.shape)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

ax1.view_init(20, -120)

ax1.plot_surface(xx, tt, e_grid, cmap='inferno', alpha=.8)

plt.tight_layout()
plt.savefig(f"{folder_path}figures/source2.png", dpi=300, bbox_inches='tight')
plt.show()