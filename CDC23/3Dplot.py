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

# Matlab's solution for the system
# dat = np.loadtxt("matlab/system/output_matlab_pde.txt")
# # x = dat[:, 0:1]
# # t = dat[:, 1:2]
# theta_true = dat[:, 2:]
# Theta_matlab = theta_true.reshape(X.shape)

# Matlab's solution for the observer
# dato = np.loadtxt("matlab/system/output_matlab_observer.txt")
# theta_hat_true = dato[:, 2:]
# Theta_hat_matlab = theta_hat_true.reshape(X.shape)

# PINNs' solution for the system
theta_pred = Ta + fun(XX)*(TM-Ta)
Theta_pinns = theta_pred.reshape(X.shape)

# Difference for the system
# Diff = Theta_matlab - Theta_pinns

# PINNs' solution for the observer
# f = np.ravel(T)
# a = len(f)
# Xo = np.vstack((np.ravel(X).reshape(a, ),
#                y2(f).reshape(a, ),
#                f.reshape(a, ))).T
# theta_hat_pred = fun_obs(Xo)
# Theta_hat_pinns = theta_hat_pred.reshape(X.shape)

# Matlab's error
# Diff_matlab = Theta_matlab - Theta_hat_matlab

# PINNs' error
# Diff_pinns = Theta_pinns - Theta_hat_pinns

# Matlab's solution for the observer at t_final
# datf = np.loadtxt("matlab/obs+syst_{}/output_f.txt".format(n))
# theta_true_f = datf[:, 0:1]
# theta_hat_true_f = datf[:, 1:]

# PINNS' prediction at t_final
# b = np.full_like(x, 1)
# c = len(x)
# Xf = np.vstack((x, b.reshape(c, ))).T
# Xof = np.vstack((x,
#                y2(b).reshape(c, ), y3(b).reshape(c, ),
#                b.reshape(c, ))).T
# theta_pred_f = fun(Xf)
# theta_hat_pred_f = fun_obs(Xof)


# Create 3D axes
fig = plt.figure()
# ax1 = fig.add_subplot(331, projection='3d')
# ax2 = fig.add_subplot(332, projection='3d')
# ax3 = fig.add_subplot(333, projection='3d')
ax4 = fig.add_subplot(131, projection='3d')
# ax5 = fig.add_subplot(132, projection='3d')
# ax6 = fig.add_subplot(133, projection='3d')

# Plot surface
# ax1.plot_surface(X, T, Theta_matlab, cmap='viridis', alpha=.8)
# ax2.plot_surface(X, T, Theta_hat_matlab, cmap='viridis', alpha=.8)
# ax3.plot_surface(X, T, Diff_matlab, cmap='viridis', alpha=.8)
ax4.plot_surface(X, 1800*T, Theta_pinns, cmap='viridis', alpha=.8)
# ax5.plot_surface(X, T, Theta_hat_pinns, cmap='viridis', alpha=.8)
# ax6.plot_surface(X, T, Diff_pinns, cmap='viridis', alpha=.8)

# Set labels and title
# ax1.set_xlabel('x')
# ax1.set_ylabel(r'$\tau$')
# ax1.set_title(r'$\theta_{matlab}$', fontsize=22)
#
# ax2.set_xlabel('x')
# ax2.set_ylabel(r'$\tau$')
# ax2.set_title(r'$\hat \theta_{matlab}$', fontsize=22)
#
# ax3.set_xlabel('x')
# ax3.set_ylabel(r'$\tau$')
# ax3.set_title(r'$e_{matlab}$', fontsize=17)
ax4.set_xlabel('Distance x')
ax4.set_ylabel(r'Time t')
ax4.set_zlabel(r'Temperature')
ax4.set_title(r'PINNs prediction of the system', fontsize=18)

plt.yticks(np.arange(0, 1801, 600.0))

# ax5.set_xlabel('x')
# ax5.set_ylabel(r'$\tau$')
# ax5.set_title(r'$\hat\theta_{pinns}$', fontsize=20)
#
# ax6.set_xlabel('x')
# ax6.set_ylabel(r'$\tau$')
# ax6.set_title(r'$e_{pinns}$', fontsize=20)
ax4.view_init(20, -120)
fig.subplots_adjust(bottom = - 1.2)
fig.subplots_adjust(right = 2)

# Save and show plot
plt.savefig("figures/system_.png",
            dpi=300, bbox_inches='tight')
plt.show()

