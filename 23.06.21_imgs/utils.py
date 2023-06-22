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
import torch

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
msg = None
ITERATION = 0
ij = 0

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
output_path = None



def inizia_esperimento(ii, cc, ss):
    print(f"Inizio esperimento:{cc, ss, ii}")
    global output_path
    #
    # if gp_seed is None:
    #     gp_seed = random.randint(0, 1000)  # Genera il seed per GP
    # if dde_seed is None:
    #     dde_seed = random.randint(0, 1000)  # Genera il seed per DDE
    #
    # ini_list = ["Glorot normal", "Glorot uniform", "He normal", "He uniform", "zeros"]
    # initial = ini_list[ij]
    #
    # # Crea il nome della cartella basato sui seed
    # output_path = f"{folder_path}output/{gp_seed}_{dde_seed}_{initial}"
    output_path = f"{folder_path}output/{ii}_{cc}_{ss}/"

    # Crea la struttura delle cartelle
    cartella_figure = f"{output_path}figures"
    cartella_history = f"{output_path}history"
    cartella_model = f"{output_path}model"

    # Crea le cartelle se non esistono già
    os.makedirs(cartella_figure, exist_ok=True)
    os.makedirs(cartella_history, exist_ok=True)
    os.makedirs(cartella_model, exist_ok=True)

    # Chiamata alla funzione per il codice specifico dell'esperimento
    # esegui_codice_specifico()

    # Restituisci i seed come output
    # return gp_seed, dde_seed, initial, ij
    return output_path


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
    dtheta_tau = dde.grad.jacobian(theta, x, i=0, j=2)
    dtheta_xx = dde.grad.hessian(theta, x, i=0, j=0)

    return a1 * dtheta_tau - dtheta_xx + a2 * theta * W_avg


def func(x):
    return x[:, 1:2] * (x[:, 0:1] ** 4) / 4 + 15 * (((x[:, 0:1] - 1) ** 2) * x[:, 0:1]) / dT


def transform(x, y):
    return x[:, 0:1] * y

def transform2(x, y):
    mask = torch.where(x[:, 2:] == 0, torch.ones_like(y), torch.zeros_like(y))
    transformed_y = mask * (x[:, 1:2] * (x[:, 0:1] ** 4) / 4 + 15 * (((x[:, 0:1] - 1) ** 2) * x[:, 0:1]) / dT) \
                    + (1 - mask) * y
    return x[:, 0:1] * transformed_y


def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def create_model(ii, config, settings):
    # global gp_seed, dde_seed, initial
    dde_seed, initial = ii
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation = config
    end_time, weight_ic, end_flux, spec = settings
    start_flux = -1* spec * end_flux

    geom = dde.geometry.Rectangle([0, start_flux], [1, end_flux])
    timedomain = dde.geometry.TimeDomain(0, end_time)
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
    # net.apply_output_transform(transform2)

    loss_weights = [1, 1, weight_ic]
    model = dde.Model(data, net)
    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def train_model(model, ii, config, ss):
    # global gp_seed, dde_seed, initial
    global output_path
    dde_seed, initial = ii
    dde.config.set_random_seed(dde_seed)

    print(f"start training {config}_{ss}")
    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{output_path}model/{config}_{ss}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{config}_{ss}_loss",
                 train_fname=f"{config}_{ss}_train", test_fname=f"{config}_{ss}_test",
                 output_dir=f"{output_path}history")
    e = scarto(model)

    # return np.linalg.norm(e)
    return model


def scarto(model):
    pinns = {'c': model.predict(XX['c']), 'l': model.predict(XX['l']),
             'e': model.predict(XX['e']), 's': model.predict(XX['s'])}

    error = [dde.metrics.l2_relative_error(matlab['c'], pinns['c']),
             dde.metrics.l2_relative_error(matlab['l'], pinns['l']),
             dde.metrics.l2_relative_error(matlab['e'], pinns['e']),
             dde.metrics.l2_relative_error(matlab['s'], pinns['s'])]
    return error

def restore_model(model, ii, config, ss):
    # global gp_seed, dde_seed, initial
    global output_path
    dde_seed, initial = ii
    dde.config.set_random_seed(dde_seed)

    print(f"restoring {config}")
    model.restore(f"{output_path}model/{config}_{ss}.ckpt-{epochs}.pt", verbose=0)
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


def plot_3d(ii, confi, sets):
    # global gp_seed, dde_seed, initial
    global output_path
    dde_seed, initial = ii
    dde.config.set_random_seed(dde_seed)

    a = create_model(ii, confi, sets)
    p = restore_model(a, ii, confi, sets)

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
    global ITERATION, gp_seed, dde_seed, initial, output_path
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
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
    global gp_seed, dde_seed, initial, output_path
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

def data_analysis(output_fold):
    # Navigate to the output folder
    os.chdir(output_fold)

    # Get the list of subdirectories in the output folder
    folders = [f for f in os.listdir(".") if os.path.isdir(f)]

    with pd.ExcelWriter("AA_RES.xlsx", engine="xlsxwriter") as writer:
        for folder in folders:
            # Create the full path to the folder
            folder_path = f"{output_fold}/{folder}"

            # Navigate to the current folder
            os.chdir(folder_path)

            # Read the hpo_results.csv file
            csv_file = "hpo_results.csv"
            a = pd.read_csv(csv_file)

            # Apply transformations to the dataframe a
            pd.options.display.float_format = '{:.2e}'.format
            a["Learning Rate"] = a["Learning Rate"].apply(lambda x: '%.2e' % x)
            a["Time Spent"] = a["Time Spent"].apply(lambda x: '%.2e' % x)
            a["Metric"] = a["Metric"].apply(lambda x: '%.2e' % x)
            a["Config"] = a[["Learning Rate", "Num Dense Layers", "Num Dense Nodes",
                             "Activation"]].apply(lambda x: '[' + ','.join(map(str, x)) + ']', axis=1)
            b = a.loc[:, ["Config", "Metric", "Error"]].sort_values(by="Error", ascending=True)
            b["Iteration"] = b.index

            # Navigate back to the output folder
            os.chdir(output_fold)
            b.to_excel(writer, sheet_name=folder, index=False)


def plot_err(ii, cc, ss):
    global output_path
    a = create_model(ii, cc, ss)
    b = restore_model(a, ii, cc, ss)

    pinns = {'c': b.predict(XX['c']), 'l': b.predict(XX['l']),
             'e': b.predict(XX['e']), 's': b.predict(XX['s'])}

    error = {'x': np.ravel(X), 't': np.ravel(T),
             'c': np.abs(matlab['c'] - pinns['c']), 'l': np.abs(matlab['l'] - pinns['l']),
             'e': np.abs(matlab['e'] - pinns['e']), 's': np.abs(matlab['s'] - pinns['s'])}

    error_values = [np.linalg.norm(error['c']), np.linalg.norm(error['l']),
                    np.linalg.norm(error['e']), np.linalg.norm(error['s'])]

    x_labels = ['constant flux', 'linear flux', 'exponential flux', 'sinusoidal flux']

    # Create the bar graph
    plt.bar(x_labels, error_values)

    # Customize labels and title
    plt.xlabel('Error')
    plt.ylabel('Value')
    plt.title('Error Values')

    # Add labels to each bar
    # for i, value in enumerate(error_values):
    #     plt.text(i, value, str(round(value, 2)), ha='center', va='bottom')

    # Display the bar graph
    plt.savefig(f"{output_path}figures/error_domain_{cc}_{ss}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Reshape the error dictionary to have the same shape as the grid
    error_grid = {}
    for key in error.keys():
        error_grid[key] = np.reshape(error[key], X.shape)

    # Calculate the error evolution in time
    error_evolution_time = {
        'Constant Flux': np.linalg.norm(error_grid['c'], axis=1),
        'Linear Flux': np.linalg.norm(error_grid['l'], axis=1),
        'Exponential Flux': np.linalg.norm(error_grid['e'], axis=1),
        'Sinusoidal Flux': np.linalg.norm(error_grid['s'], axis=1)
    }

    # Calculate the error in space
    error_in_space = {
        'Constant Flux': np.linalg.norm(error_grid['c'], axis=0),
        'Linear Flux': np.linalg.norm(error_grid['l'], axis=0),
        'Exponential Flux': np.linalg.norm(error_grid['e'], axis=0),
        'Sinusoidal Flux': np.linalg.norm(error_grid['s'], axis=0)
    }

    # Plot error evolution in time
    plt.figure(figsize=(10, 8))
    for i, (key, value) in enumerate(error_evolution_time.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(t, value)
        plt.title(key)
        plt.xlabel('Time')
        plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig(f"{output_path}figures/error_time_{cc}_{ss}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot error in space
    plt.figure(figsize=(10, 8))
    for i, (key, value) in enumerate(error_in_space.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(x, value)
        plt.title(key)
        plt.xlabel('Space')
        plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig(f"{output_path}figures/error_space_{cc}_{ss}.png", dpi=300, bbox_inches='tight')
    plt.show()


    # Save the data as an XLSX file
    data = {'Constant Flux': [np.linalg.norm(error_grid['c']), np.sum(np.linalg.norm(error_grid['c'], axis=0)), np.sum(np.linalg.norm(error_grid['c'], axis=1))],
            'Linear Flux': [np.linalg.norm(error_grid['l']), np.sum(np.linalg.norm(error_grid['l'], axis=0)), np.sum(np.linalg.norm(error_grid['l'], axis=1))],
            'Exponential Flux': [np.linalg.norm(error_grid['e']), np.sum(np.linalg.norm(error_grid['e'], axis=0)), np.sum(np.linalg.norm(error_grid['e'], axis=1))],
            'Sinusoidal Flux': [np.linalg.norm(error_grid['s']), np.sum(np.linalg.norm(error_grid['s'], axis=0)), np.sum(np.linalg.norm(error_grid['s'], axis=1))]}
    
    df = pd.DataFrame(data, index=['Domain', 'Space', 'Time'])
    df.to_excel(f"{output_path}/error.xlsx")

