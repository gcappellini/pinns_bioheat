import utils2
from deepxde.geometry.geometry_3d import Sphere, Cuboid
from deepxde.geometry.csg import CSGDifference, CSGUnion, CSGIntersection
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

# Define the dimensions of the cylinder
radius = 0.7  # Radius of the cylinder
height = 1.0  # Height of the cylinder

# Create the sphere for the curved part of the cylinder
sphere = Sphere(center=(0.5, 0.5, 0.5), radius=radius)

# Create the rectangle for the flat part of the cylinder
rectangle = Cuboid(xmin=[0, 0, 0.5], xmax=[1.0, 1.0, 1.0])
rectangle2 = Cuboid(xmin=[0, 0, 0], xmax=[1.0, 1.0, 0.5])

# Perform a union operation to combine the sphere and rectangle into a cylinder
emysphere = CSGIntersection(sphere, rectangle)
geom = CSGUnion(emysphere, rectangle2)

X0 = geom.random_points(5000)
y = utils2.source(torch.Tensor(X0)).cpu()
# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, and z coordinates from X0t
x_coords = X0[:, 0]
y_coords = X0[:, 1]
z_coords = X0[:, 2]

# Plot the output 'sys' at the 3D coordinates
sc = ax.scatter(x_coords, y_coords, z_coords, c=y, cmap='viridis') # , marker='o')

# Add a color bar to the plot
plt.colorbar(sc)

# Set labels for the axes
ax.set_title(f'Source')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.savefig(f"{folder_path}figures/source.png", dpi=300, bbox_inches='tight')
plt.show()
e = np.array(y)
print(np.max(e), np.min(e))