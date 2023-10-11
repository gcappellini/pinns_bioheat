import numpy as np
import matplotlib.pyplot as plt

# Define convex function
def convex(a):
    return 1 - 0.2 * (a**2)

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
x, y = np.meshgrid(x, y)
z_upper = convex(x)
z_lower = -np.ones_like(z_upper)

# Make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the convex upper and lower surfaces
ax.plot_surface(x, y, z_upper, alpha=0.9, edgecolor='k', rstride=100, cstride=100)
ax.plot_surface(x, y, z_lower, alpha=0.9, edgecolor='k', rstride=100, cstride=100)

y = np.linspace(-1, 1, 10)
z = np.linspace(-1, convex(1), 10)
y, z = np.meshgrid(y, z)
x_right = np.ones_like(y)
x_left = -np.ones_like(y)

ax.plot_surface(x_right, y, z, alpha=0.9, edgecolor='k', rstride=100, cstride=100)
ax.plot_surface(x_left, y, z, alpha=0.9, edgecolor='k', rstride=100, cstride=100)

# x = np.linspace(-1, 1, 10)
# z = np.linspace(-1, convex(x), 10)
# xv = np.outer(x, np.ones_like(x))
# y_right = np.ones_like(xv)
# y_left = -np.ones_like(xv)

# ax.plot_surface(xv, y_right, z, alpha=0.9, edgecolor='k', rstride=100, cstride=100)
# ax.plot_surface(xv, y_left, z, alpha=0.9, edgecolor='k', rstride=100, cstride=100)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
