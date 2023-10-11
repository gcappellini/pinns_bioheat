import numpy as np
import matplotlib.pyplot as plt

# Define convex function
def convex_z(x, y):
    return 1 - 0.2 * (x**2 + y**2)

# Create grid and multivariate normal
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
x, y = np.meshgrid(x, y)
z_upper = convex_z(x, y)
z_lower = -np.ones_like(z_upper)

# Make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the convex upper surface
ax.plot_surface(x, y, z_upper, alpha=0.9, edgecolor='k', rstride=100, cstride=100)

# Plotting the lower surface
ax.plot_surface(x, y, z_lower, alpha=0.9, edgecolor='k', rstride=100, cstride=100)

ax.plot_surface(x, -np.ones_like(x), y, alpha=0.9, edgecolor='k', rstride=100, cstride=100)
ax.plot_surface(x, np.ones_like(x), y, alpha=0.9, edgecolor='k', rstride=100, cstride=100)

ax.plot_surface(-np.ones_like(x), x, y, alpha=0.9, edgecolor='k', rstride=100, cstride=100)
ax.plot_surface(np.ones_like(x), x, y, alpha=0.9, edgecolor='k', rstride=100, cstride=100)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
