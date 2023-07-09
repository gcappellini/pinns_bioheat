import numpy as np
from system import fun, y2
from observer import fun_1obs
import deepxde as dde
import matplotlib.pyplot as plt


TM = 45
Ta = 37

# Grid
x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)
XX = np.vstack((np.ravel(X),
               np.ravel(T))).T

# PINNs' solution for the system
theta_pred = Ta + fun(XX)*(TM-Ta)
Theta_pinns = theta_pred.reshape(X.shape)

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
ax4.set_title(r'PINNs prediction of the system', fontsize=8, y=.96, weight='semibold')

plt.yticks(np.arange(0, 1801, 600.0), fontsize=7)
plt.xticks(fontsize=7)
ax4.tick_params(axis='z', labelsize=7)


ax4.view_init(20, -120)

# Save and show plot
plt.savefig("figures/system.png",
            dpi=300, bbox_inches='tight')
plt.show()

