import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define a function for the convex surface
def convex_surface(x, y):
    return 1 - 0.2 * (x**2 + y**2)

# Generate grid points
x = np.linspace(-1, 1, 30)
y = np.linspace(-1, 1, 30)
x, y = np.meshgrid(x, y)
z_upper = convex_surface(x, y)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the convex upper surface
ax.plot_surface(x, y, z_upper, alpha=0.7, color='c', rstride=1, cstride=1, edgecolor='none')

# Define vertices of the box sides
zs = np.array([z_upper, np.full_like(z_upper, -1)])
zs_min = zs.min()
zs_max = zs.max()

vertices = [
    [(x[i, 0], y[i, 0], z_upper[i, 0]) for i in range(x.shape[0])] + [(x[i, 0], y[i, 0], zs_min) for i in range(x.shape[0]-1, -1, -1)],
    [(x[i, -1], y[i, -1], z_upper[i, -1]) for i in range(x.shape[0])] + [(x[i, -1], y[i, -1], zs_min) for i in range(x.shape[0]-1, -1, -1)],
    [(x[0, i], y[0, i], z_upper[0, i]) for i in range(x.shape[1])] + [(x[0, i], y[0, i], zs_min) for i in range(x.shape[1]-1, -1, -1)],
    [(x[-1, i], y[-1, i], z_upper[-1, i]) for i in range(x.shape[1])] + [(x[-1, i], y[-1, i], zs_min) for i in range(x.shape[1]-1, -1, -1)],
]

# Plotting the sides of the box
ax.add_collection3d(Poly3DCollection(vertices, facecolors='c', linewidths=1, edgecolors='gray', alpha=0.2))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Box with Convex Upper Surface')

plt.show()
