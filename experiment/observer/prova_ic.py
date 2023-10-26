import numpy as np
import os
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

k = 1

def ic_obs(x):
    y1 = x[0, 1:2]
    y2 = x[0, 2:3]
    y3 = x[0, 3:4]
    b1 =  6# arbitrary parameter
    # a2 = 0
    # e = y1 + (y3 - 2*a2 + k*(y2-y1-a2))*x + a2*x**2
    e = (y3 + k * (y2 - y1))/(b1 * np.cos(b1) + k * np.sin(b1))* np.sin(b1*x) + y1
    return e[:, 0:1]

def gen_obsdata():
    data = np.loadtxt(f"{folder_path}output_matlab_pde.txt")
    x = data[:, 0:1]
    x_end = np.min(x)
    theta_end = data[data[:, 0] == x_end]

    unique_x = np.unique(x)
    sorted_unique_x = np.sort(unique_x)[::-1]
    x_sup = sorted_unique_x[1]
    theta_sup = data[data[:, 0] == x_sup]

    x_max = np.max(x)
    theta_extra = data[data[:, 0] == x_max]
    fl = (theta_extra[:, 2:] - theta_sup[:, 2:])/(x_max - x_sup)

    mask = data[:, 0] != x_max
    data = data[mask, :]
    x, t, exact = data[:, 0:1], data[:, 1:2], data[:, 2:]
    y = exact.flatten()[:, None]
    num_rows = data.shape[0]
    new_columns = np.zeros((num_rows, 3))
    X = np.hstack((x, new_columns, t))

    unique_t = np.unique(t)
    # for el in range(len(unique_t)):
    for el in [0]:
        a = X[X[:, 4] == unique_t[el]]
        a[:, 1] = theta_end[el, 2:]
        a[:, 2] = theta_sup[el, 2:]
        a[:, 3] = fl[el]
        X[X[:, 4] == unique_t[el]] = a

    return X, y


a, b = gen_obsdata()
a0 = a[a[:, 4]==0]
y1 = b[:len(a0), :]
y2 = ic_obs(a0)

# Create a plot
plt.plot(a0[:, 0:1], y1, label='y1', marker='s', linestyle='--', markevery=5)
plt.plot(a0[:, 0:1], y2, label='y2', marker='o', linestyle='-', markevery=7)
plt.legend()
plt.savefig(f"{folder_path}figures/prova_ic.png", dpi=300, bbox_inches='tight')
# Display the plot
plt.show()
