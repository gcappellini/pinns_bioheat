import os
import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import pandas as pd
from openpyxl import load_workbook
import time
import random
import imgkit

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin


gp_seed = None
dde_seed = None
initial = None
ITERATION = 0
ij = 0

folder_path = "/home/giuglielmocappellini/Projects/PINNs/23.06.14/"
# folder_path = ""

def esegui_esperimento():
    global gp_seed, dde_seed, initial, ij

    if gp_seed is None:
        gp_seed = random.randint(0, 1000)  # Genera il seed per GP
    if dde_seed is None:
        dde_seed = random.randint(0, 1000)  # Genera il seed per DDE

    ini_list = ["Glorot normal", "Glorot uniform", "He normal", "He uniform", "zeros"]
    initial = ini_list[ij]

    # Crea il nome della cartella basato sui seed
    output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}"

    # Crea la struttura delle cartelle
    cartella_figure = os.path.join(output_path, "figures")
    cartella_history = os.path.join(output_path, "history")
    cartella_model = os.path.join(output_path, "model")

    # Crea le cartelle se non esistono già
    os.makedirs(cartella_figure, exist_ok=True)
    os.makedirs(cartella_history, exist_ok=True)
    os.makedirs(cartella_model, exist_ok=True)

    # Chiamata alla funzione per il codice specifico dell'esperimento
    # esegui_codice_specifico()

    # Restituisci i seed come output
    return gp_seed, dde_seed, initial, ij



# General parameters
L0 = 0.05
TM = 45
Ta = 37
tauf = 1800

rho = 1050
c = 3639
k_eff = 5
alfa = rho * c / k_eff

W_avg = 2.3
W_min = 0.45
W_max = 4
cb = 3825

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff

q0 = 16
dT = TM - Ta
q0_ad = q0/dT

# Network parameters
precision_train = 10
precision_test = 30
epochs = 20000

# HPO setting
n_calls = 100
dim_learning_rate = Real(low=1e-5, high=5e-1, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=4, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"],
                             name="activation")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation
]

# Prediction grid and ground truth
x = np.linspace(0, 1, 101)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)


def rescale(theta):
    return Ta + theta*(TM-Ta)

def cost(t):
    return np.full_like(t, q0_ad)


def lin(t):
    return q0_ad*(1-t)


def expo(t):
    return q0_ad*np.exp(-t)


def sinu(t):
    return q0_ad*np.cos(np.pi*t/2)


functions = {
    'c': cost,
    'l': lin,
    'e': expo,
    's': sinu
}

XX = {
    key: np.vstack((np.ravel(X), func(np.ravel(T)), np.ravel(T))).T
    for key, func in functions.items()
}

matlab = {'c': np.loadtxt(f"{folder_path}matlab/output_matlab_system_0.txt")[:, 2:],
          'l': np.loadtxt(f"{folder_path}matlab/output_matlab_system_1.txt")[:, 2:],
          'e': np.loadtxt(f"{folder_path}matlab/output_matlab_system_2.txt")[:, 2:],
          's': np.loadtxt(f"{folder_path}matlab/output_matlab_system_3.txt")[:, 2:]}

def pde(x, theta):
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

    return a1 * dtheta_tau - dtheta_xx + a2 * theta * W_avg


def func(x):
    return x[:, 1:2] * (x[:, 0:1] ** 4) / 4 + 15 * (((x[:, 0:1] - 1) ** 2) * x[:, 0:1]) / dT


def transform(x, y):
    return x[:, 0:1] * y


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def create_model(config):
    global gp_seed, dde_seed, initial
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation = config

    geom = dde.geometry.Rectangle([0, -5], [1, 5])
    timedomain = dde.geometry.TimeDomain(0, 1.3)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: x[:, 1:2], boundary_1)

    ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_1, ic],
        num_domain=2560,
        num_boundary=80,
        num_initial=160,
        num_test=2560,
    )

    net = dde.maps.FNN(
        [3] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initial,
    )

    net.apply_output_transform(transform)

    loss_weights = [1, 1, 10000]
    model = dde.Model(data, net)
    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def train_model(model, config):
    global gp_seed, dde_seed, initial
    output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}/"
    dde.config.set_random_seed(dde_seed)

    print(f"start training {config}")
    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{output_path}model/{config}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname=f"{config}_loss",
                 train_fname=f"{config}_train", test_fname=f"{config}_test",
                 output_dir=f"{output_path}history")

    pinns = {'c': model.predict(XX['c']), 'l': model.predict(XX['l']),
             'e': model.predict(XX['e']), 's': model.predict(XX['s'])}

    error = [dde.metrics.l2_relative_error(matlab['c'], pinns['c']),
             dde.metrics.l2_relative_error(matlab['l'], pinns['l']),
             dde.metrics.l2_relative_error(matlab['e'], pinns['e']),
             dde.metrics.l2_relative_error(matlab['s'], pinns['s'])]

    e = np.linalg.norm(error)
    return e


def restore_model(model, config):
    global gp_seed, dde_seed, initial
    output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}/"
    dde.config.set_random_seed(dde_seed)

    print(f"restoring {config}")
    model.restore(f"{output_path}model/{config}.ckpt-{epochs}.pt", verbose=0)
    return model
    # pinns = {'c': model.predict(XX['c']), 'l': model.predict(XX['l']),
    #          'e': model.predict(XX['e']), 's': model.predict(XX['s'])}
    #
    # error = [dde.metrics.l2_relative_error(matlab['c'], pinns['c']),
    #          dde.metrics.l2_relative_error(matlab['l'], pinns['l']),
    #          dde.metrics.l2_relative_error(matlab['e'], pinns['e']),
    #          dde.metrics.l2_relative_error(matlab['s'], pinns['s'])]
    #
    # e = np.linalg.norm(error)
    # return e

def configure_subplot(ax, surface):
    ax.plot_surface(X, 1800 * T, surface, cmap='inferno', alpha=.8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)


def plot_3d(confi):
    global gp_seed, dde_seed, initial
    output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}/"
    dde.config.set_random_seed(dde_seed)

    a = create_model(confi)
    p = restore_model(a, confi)

    pinns = {'c': p.predict(XX['c']), 'l': p.predict(XX['l']),
             'e': p.predict(XX['e']), 's': p.predict(XX['s'])}

    # Create 3D axes
    fig = plt.figure(figsize=(9, 12))

    # Define row and column titles
    row_titles = ['Constant Flux', 'Linear Flux', 'Exponential Flux', 'Sinusoidal Flux']
    col_titles = ['MATLAB', 'PINNs', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [rescale(matlab['c']).reshape(X.shape), rescale(pinns['c']).reshape(X.shape),
         np.abs(pinns['c'] - matlab['c']).reshape(X.shape)],
        [rescale(matlab['l']).reshape(X.shape), rescale(pinns['l']).reshape(X.shape),
         np.abs(pinns['l'] - matlab['l']).reshape(X.shape)],
        [rescale(matlab['e']).reshape(X.shape), rescale(pinns['e']).reshape(X.shape),
         np.abs(pinns['e'] - matlab['e']).reshape(X.shape)],
        [rescale(matlab['s']).reshape(X.shape), rescale(pinns['s']).reshape(X.shape),
         np.abs(pinns['s'] - matlab['s']).reshape(X.shape)]
    ]

    # Create a grid of subplots
    grid = plt.GridSpec(4, 3)

    # Iterate over rows and columns to add subplots
    for row in range(4):
        for col in range(3):
            ax = fig.add_subplot(grid[row, col], projection='3d')
            configure_subplot(ax, surfaces[row][col])

            # Set column titles for the top row
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8, y=.96, weight='semibold')

            # Set row titles for the leftmost column
            if col == 0:
                ax.text2D(-0.2, 0.5, row_titles[row], fontsize=9, rotation='vertical', ha='center',
                          va='center', weight='semibold', transform=ax.transAxes)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # Save and show plot
    plt.savefig(f"{output_path}figures/plot3d_{confi}.png", dpi=300, bbox_inches='tight')
    plt.show()


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    global ITERATION, gp_seed, dde_seed, initial
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
    output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}/"
    dde.config.set_random_seed(dde_seed)

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print()

    start_time = time.time()

    # Create the neural network with these hyper-parameters.
    mm = create_model(config)
    # possibility to change where we save
    error = train_model(mm, config)
    # print(accuracy, 'accuracy is')

    if np.isnan(error):
        error = 10 ** 5

    end_time = time.time()
    time_spent = end_time - start_time

    # Store the configuration and error in a DataFrame
    data = {
        "Learning Rate": learning_rate,
        "Num Dense Layers": num_dense_layers,
        "Num Dense Nodes": num_dense_nodes,
        "Activation": activation,
        "Error": error,
        "Time Spent": time_spent
    }
    df = pd.DataFrame(data, index=[ITERATION])
    df["Metric"] = df.apply(metric, axis=1)

    file_path = f"{output_path}hpo_results.csv"

    if not os.path.isfile(file_path):
        # Create a new CSV file with the header
        df.to_csv(file_path, index=False)
    else:
        # Append the DataFrame to the CSV file
        df.to_csv(file_path, mode='a', header=False, index=False)

    ITERATION += 1
    return error


def hpo(default_parameters):
    global gp_seed, dde_seed, initial
    output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}/"
    dde.config.set_random_seed(dde_seed)

    search_result = gp_minimize(
        func=fitness,
        dimensions=dimensions,
        acq_func="EI",  # Expected Improvement.
        n_calls=n_calls,
        x0=default_parameters,
        random_state=gp_seed,
    )

    print(search_result.x)

    # Plot convergence and save the figure
    plt.figure()
    plot_convergence(search_result)
    plt.title('Convergence Plot')
    plt.savefig(f"{output_path}figures/plot_conv.png",
            dpi=300, bbox_inches='tight')

    # Plot objective and save the figure
    plt.figure()
    plot_objective(search_result, show_points=True, size=3.8)
    plt.savefig(f"{output_path}figures/plot_obj.png",
                dpi=300, bbox_inches='tight')

    plt.show()

    # Load the CSV file into a pandas DataFrame
    csv_file = f'{output_path}hpo_results.csv'
    a = pd.read_csv(csv_file)
    a.to_excel(f'{output_path}data.xlsx', index=False)
    os.remove(csv_file)

    # set the display format for floats to scientific notation
    pd.options.display.float_format = '{:.2e}'.format

    # format the desired columns
    a["Learning Rate"] = a["Learning Rate"].apply(lambda x: '%.2e' % x)
    # a["Error"] = a["Error"].apply(lambda x: '%.2e' % x)
    a["Time Spent"] = a["Time Spent"].apply(lambda x: '%.2e' % x)
    a["Metric"] = a["Metric"].apply(lambda x: '%.2e' % x)

    a["Config"] = a[["Learning Rate", "Num Dense Layers", "Num Dense Nodes",
                       "Activation"]].apply(lambda x: '[' + ','.join(map(str, x)) + ']', axis=1)

    b = a.loc[:, ["Config", "Metric", "Error"]].sort_values(by="Error", ascending=True)

    b.to_excel(f"{output_path}list.xlsx", index=False)

    return search_result.x


def metric(row):
    L_1 = row['Num Dense Layers']
    N = row['Num Dense Nodes']
    N0 = 3
    net = np.ones(L_1+2) * N
    net[0] = N0
    net[L_1+1] = 1

    result = np.dot(net[:-1]+1, net[1:])
    return result

def reset_iteration():
    global ITERATION, gp_seed, dde_seed, initial, ij
    ITERATION = 0
    gp_seed = None
    dde_seed = None
    initial = None

    if ij == 4:
        ij = 0
    else:
        ij += 1
