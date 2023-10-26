import os
import numpy as np

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

data = np.loadtxt(f"{folder_path}output_matlab_pde.txt")
x, t, exact = data[:, 0:1], data[:, 1:2], data[:, 2:]
y = exact.flatten()[:, None]

x_end = np.min(x)
theta_end = data[x[:, 0] == x_end]

unique_x = np.unique(x)
sorted_unique_x = np.sort(unique_x)[::-1]
x_sup = sorted_unique_x[1]
theta_sup = data[x[:, 0] == x_sup]
theta_sup[:, 0] = 1

x_max = np.max(x)
theta_extra = data[x[:, 0] == x_max]
fl = (theta_extra[:, 2:] - theta_sup[:, 2:])/(x_max - x_sup)

num_rows = data.shape[0]
new_columns = np.zeros((num_rows, 3))
X = np.hstack((x, new_columns, t))

unique_t = np.unique(t)
for el in range(len(unique_t)):
    a = X[X[:, 4] == unique_t[el]]
    a[:, 1] = theta_end[el, 2:]
    a[:, 2] = theta_sup[el, 2:]
    a[:, 3] = fl[el]
    X[X[:, 4] == unique_t[el]] = a

print(X)


# X = np.vstack((x, theta_sup, fl, t)).T



