import numpy as np
from system import fun, y2
from observer import fun_1obs
import deepxde as dde
import matplotlib.pyplot as plt

def gen_testdata():
    data = np.loadtxt(f"matlab/system/output_matlab_pde.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y

x, matlab = gen_testdata()

ll = int(np.sqrt(len(matlab)))

pinns = fun(x)

TM = 45
Ta = 37

# Grid
x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)
# XX = np.vstack((np.ravel(X),
#                np.ravel(T))).T

# PINNs' solution for the system
theta_pred = Ta + pinns*(TM-Ta)
Theta_pinns = theta_pred.reshape(ll, ll)

theta_true = Ta + matlab*(TM-Ta)
Theta_matlab = theta_true.reshape(ll, ll)

err = np.abs(Theta_pinns-Theta_matlab)

# Create 3D axes
fig = plt.figure()

ax4 = fig.add_subplot(111, projection='3d')


# Plot surface
ax4.plot_surface(X, 1800*T, Theta_pinns, cmap='viridis', alpha=.8)

# Get rid of the panes
ax4.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax4.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax4.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# Set labels and title
ax4.set_xlabel('Distance x', fontsize=7)
ax4.set_ylabel(r'Time t', fontsize=7)
ax4.set_zlabel(r'Temperature', fontsize=7)
# ax4.set_title(r'Error Matlab vs. PINNs', fontsize=8, y=.96, weight='semibold')

plt.yticks(np.arange(0, 1801, 600.0), fontsize=7)
plt.xticks(fontsize=7)
ax4.tick_params(axis='z', labelsize=7)


ax4.view_init(20, -120)

# Save and show plot
plt.savefig("figures/system2.png",
            dpi=300, bbox_inches='tight')
plt.show()

