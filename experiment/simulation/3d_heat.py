"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import utils2
import matplotlib.pyplot as plt


# Problem parameters:
a = 0.4  # Thermal diffusivity
L = 1  # Length of the bar
n = 1  # Frequency of the sinusoidal initial conditions

def pde(x, y):
    """Expresses the PDE residual of the heat equation."""
    dy_t = dde.grad.jacobian(y, x, i=0, j=3)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    dy_zz = dde.grad.hessian(y, x, i=2, j=2)
    return dy_t - a * (dy_xx + dy_xx + dy_xx) + utils2.source(x)


# Computational geometry:
geom = dde.geometry.Sphere([0, 0, 0], L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.icbc.NeumannBC(geomtime, lambda x: -1, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: 0,
    lambda _, on_initial: on_initial,
)

# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)
net = dde.nn.FNN([4] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Build and train the model:
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=2000)

# Plot/print the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
X0t = geomtime.uniform_points(5)
sys = model.predict(X0t)
print(sys)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, and z coordinates from X0t
x_coords = X0t[:, 0]
y_coords = X0t[:, 1]
z_coords = X0t[:, 2]

# Plot the output 'sys' at the 3D coordinates
sc = ax.scatter(x_coords, y_coords, z_coords, c=sys, cmap='viridis') # , marker='o')

# Add a color bar to the plot
plt.colorbar(sc)

# Set labels for the axes
ax.set_title(f'System at t={time_inst}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.savefig(f"{folder_path}figures/plot_t={time_inst}.png", dpi=300, bbox_inches='tight')
plt.show()