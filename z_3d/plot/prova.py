import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, combinations

# Sample data
time_steps = 10  # For example, you have 10 time steps

# Define box dimensions
x_min, x_max = -1, 1
y_min, y_max = -1, 1
z_min, z_max = -1, 1
n_points = 5  # Number of points along each axis

# Generate evenly distributed points within the box
x = np.linspace(x_min, x_max, n_points)
y = np.linspace(y_min, y_max, n_points)
z = np.linspace(z_min, z_max, n_points)

# Creating a grid of points
X, Y, Z = np.meshgrid(x, y, z)
points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Replicating for all time steps and generating corresponding temperatures
X = np.tile(points, (time_steps, 1, 1))
y_pred = np.random.rand(time_steps, n_points ** 3)  # Corresponding temperatures for each point at each time step


def plot_temperature_distribution_for_time(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get data for time t
    x, y, z = X[t, :, 0], X[t, :, 1], X[t, :, 2]
    temp = y_pred[t]

    # Draw points
    sc = ax.scatter(x, y, z, c=temp, cmap='viridis', s=50)
    plt.colorbar(sc)

    # Draw semi-transparent parallelepiped
    for s, e in combinations(np.array(list(product([x_min, x_max], [y_min, y_max], [z_min, z_max]))), 2):
        if np.sum(np.abs(s - e)) == x_max - x_min or np.sum(np.abs(s - e)) == y_max - y_min or np.sum(
                np.abs(s - e)) == z_max - z_min:
            ax.plot3D(*zip(*[s, e]), color="k")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f'Temperature distribution at time {t}')
    plt.show()


# Example: To visualize the temperature distribution at time t=5
plot_temperature_distribution_for_time(5)
