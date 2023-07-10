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
import wandb

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf

    sin = tf.sin


gp_seed = None
dde_seed = None
ITERATION = 0

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
# folder_path = f"{folder_pa}/"
folder_path = ""
output_path = None

epochs = 20000

def inizia_hpo():
    global output_path, gp_seed, dde_seed

    if gp_seed is None:
        gp_seed = random.randint(0, 1000)  # Genera il seed per GP
    if dde_seed is None:
        dde_seed = random.randint(0, 1000)  # Genera il seed per DDE

    output_path = f"{folder_path}hpo/{dde_seed}_{gp_seed}/"

    # Crea la struttura delle cartelle
    cartella_figure = f"{output_path}figures"
    cartella_history = f"{output_path}history"
    cartella_model = f"{output_path}model"

    # Crea le cartelle se non esistono già
    os.makedirs(cartella_figure, exist_ok=True)
    os.makedirs(cartella_history, exist_ok=True)
    os.makedirs(cartella_model, exist_ok=True)

    return output_path, gp_seed, dde_seed


def inizia_esperimento(ii, cc, ss):
    global output_path, dde_seed

    dde_seed = ii
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
    return output_path, dde_seed


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

# HPO setting
n_calls = 100
dim_learning_rate = Real(low=1e-5, high=5e-1, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=4, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
dim_activation = Categorical(categories=["elu", "relu", "selu", "silu", "sigmoid", "sin", "swish", "tanh"],
                             name="activation")
dim_initialization = Categorical(categories=["Glorot normal", "Glorot uniform", "He normal", "He uniform"],
                             name="initialization")
# dim_weight_ic = Integer(low=1, high=1000000, name="weight_ic")
# dim_weight_bcl = Integer(low=1, high=1000000, name="weight_bcl")
# dim_weight_bcr = Integer(low=1, high=1000000, name="weight_bcr")
# dim_weight_domain = Integer(low=1, high=1000000, name="weight_domain")


dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
    dim_initialization
    # dim_weight_ic,
    # dim_weight_bcl,
    # dim_weight_bcr,
    # dim_weight_domain
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


def ic_func(x):
    return x[:, 1:2] * (x[:, 0:1] ** 4) / 4 + 15 * (((x[:, 0:1] - 1) ** 2) * x[:, 0:1]) / dT



def boundary_1(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def boundary_0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def create_model(config, setts):
    global dde_seed
    dde.config.set_random_seed(dde_seed)

    learning_rate, num_dense_layers, num_dense_nodes, activation, initialization = config
    end_flux, end_time, w_domain, w_bcl, w_bcr, w_ic = setts 

    geom = dde.geometry.Rectangle([0, 0], [1, end_flux])
    timedomain = dde.geometry.TimeDomain(0, end_time)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_0)
    bc_1 = dde.icbc.NeumannBC(geomtime, lambda x: x[:, 1:2], boundary_1)

    ic = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_0, bc_1, ic],
        num_domain=2560,
        num_boundary=80,
        num_initial=160,
        num_test=2560,
    )

    net = dde.maps.FNN(
        [3] + [num_dense_nodes] * num_dense_layers + [1],
        activation,
        initialization,
    )


    loss_weights = [w_domain, w_bcl, w_bcr, w_ic]
    model = dde.Model(data, net)
    model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    return model


def train_model(model, config, setts):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    print(f"start training {config}_{setts}")
    losshistory, train_state = model.train(iterations=epochs,
                                           model_save_path=f"{output_path}model/{config}_{setts}.ckpt")
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname=f"{config}_{setts}_loss",
                 train_fname=f"{config}_{setts}_train", test_fname=f"{config}_{setts}_test",
                 output_dir=f"{output_path}history")
    
    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()

    # error = test.min()
    error = scarto(model)
    # return error
    return model

def scarto(model):
    pinns = {'c': model.predict(XX['c']), 'l': model.predict(XX['l']),
             'e': model.predict(XX['e']), 's': model.predict(XX['s'])}

    error = {'c': np.abs(matlab['c'] - pinns['c']), 'l': np.abs(matlab['l'] - pinns['l']),
             'e': np.abs(matlab['e'] - pinns['e']), 's': np.abs(matlab['s'] - pinns['s'])}

    error_values = [np.linalg.norm(error['c']), np.linalg.norm(error['l']),
                    np.linalg.norm(error['e']), np.linalg.norm(error['s'])]

    error_domain = [np.linalg.norm(np.abs(matlab['c'] - pinns['c'])),
                    np.linalg.norm(np.abs(matlab['l'] - pinns['l'])),
                    np.linalg.norm(np.abs(matlab['e'] - pinns['e'])),
                    np.linalg.norm(np.abs(matlab['s'] - pinns['s']))]

    # Reshape the error dictionary to have the same shape as the grid
    error_grid = {}
    for key in error.keys():
        error_grid[key] = np.reshape(error[key], X.shape)

    # Calculate the error evolution in time
    error_time = {
        'Constant Flux': np.linalg.norm(error_grid['c'], axis=1),
        'Linear Flux': np.linalg.norm(error_grid['l'], axis=1),
        'Exponential Flux': np.linalg.norm(error_grid['e'], axis=1),
        'Sinusoidal Flux': np.linalg.norm(error_grid['s'], axis=1)
    }

    # Calculate the error in space
    error_space = {
        'Constant Flux': np.linalg.norm(error_grid['c'], axis=0),
        'Linear Flux': np.linalg.norm(error_grid['l'], axis=0),
        'Exponential Flux': np.linalg.norm(error_grid['e'], axis=0),
        'Sinusoidal Flux': np.linalg.norm(error_grid['s'], axis=0)
    }

    # values = list(error_space.values())
    
    # return np.linalg.norm(values)
    return np.linalg.norm(error_domain)

def restore_model(model, config, setts):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    model.restore(f"{output_path}model/{config}_{setts}.ckpt-{epochs}.pt", verbose=0)
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

def restore2_model(model, config, setts):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    model.restore(f"{output_path}model/final.ckpt-{epochs}.pt", verbose=0)
    return model

def configure_subplot(ax, surface):
    ax.plot_surface(X, 1800 * T, surface, cmap='inferno', alpha=.8)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)


def plot_3d(p, confi, setts):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    pinns = {'c': p.predict(XX['c']), 'l': p.predict(XX['l']),
             'e': p.predict(XX['e']), 's': p.predict(XX['s'])}

    # Create 3D axes
    fig = plt.figure(figsize=(9, 12))

    # Define row and column titles
    row_titles = ['Constant Flux', 'Linear Flux', 'Exponential Flux', 'Sinusoidal Flux']
    col_titles = ['MATLAB', 'PINNs', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [matlab['c'].reshape(X.shape), pinns['c'].reshape(X.shape),
         np.abs(pinns['c'] - matlab['c']).reshape(X.shape)],
        [matlab['l'].reshape(X.shape), pinns['l'].reshape(X.shape),
         np.abs(pinns['l'] - matlab['l']).reshape(X.shape)],
        [matlab['e'].reshape(X.shape), pinns['e'].reshape(X.shape),
         np.abs(pinns['e'] - matlab['e']).reshape(X.shape)],
        [matlab['s'].reshape(X.shape), pinns['s'].reshape(X.shape),
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
    plt.savefig(f"{output_path}figures/plot3d_{confi}_{setts}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot2_3d(p, confi, setts):
    global output_path, dde_seed
    dde.config.set_random_seed(dde_seed)

    pinns = {'c': p.predict(XX['c']), 'l': p.predict(XX['l']),
             'e': p.predict(XX['e']), 's': p.predict(XX['s'])}

    # Create 3D axes
    fig = plt.figure(figsize=(9, 12))

    # Define row and column titles
    row_titles = ['Constant Flux', 'Linear Flux', 'Exponential Flux', 'Sinusoidal Flux']
    col_titles = ['MATLAB', 'PINNs', 'Error']

    # Define surfaces for each subplot
    surfaces = [
        [matlab['c'].reshape(X.shape), pinns['c'].reshape(X.shape),
         np.abs(pinns['c'] - matlab['c']).reshape(X.shape)],
        [matlab['l'].reshape(X.shape), pinns['l'].reshape(X.shape),
         np.abs(pinns['l'] - matlab['l']).reshape(X.shape)],
        [matlab['e'].reshape(X.shape), pinns['e'].reshape(X.shape),
         np.abs(pinns['e'] - matlab['e']).reshape(X.shape)],
        [matlab['s'].reshape(X.shape), pinns['s'].reshape(X.shape),
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
    plt.savefig(f"{output_path}figures/plot3d_final.png", dpi=300, bbox_inches='tight')
    plt.show()

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation, initialization):
    global ITERATION, gp_seed, dde_seed, output_path
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation, initialization]
    dde.config.set_random_seed(dde_seed)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="hpo-no-source",

        # track hyperparameters and run metadata
        config={
            "learning rate": learning_rate,
            "num_dense_layers": num_dense_layers,
            "num_dense_nodes": num_dense_nodes,
            "activation": activation,
            "initialization": initialization
            # "weight_ic": weight_ic,
            # "weight_bcl": weight_bcl,
            # "weight_bcr": weight_bcr,
            # "weight_domain": weight_domain        
        }
    )



    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print("initialization:", initialization)
    # print("weight_ic:", weight_ic)
    # print("weight_bcl:", weight_bcl)
    # print("weight_bcr:", weight_bcr)
    # print("weight_domain:", weight_domain)
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
        "Initialization": initialization,
        # "Weight_ic": weight_ic,
        # "Weight_bcl": weight_bcl,
        # "Weight_bcr": weight_bcr,
        # "Weight_domain": weight_domain,
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

    wandb.log({"err": error})
    wandb.finish()

    ITERATION += 1
    return error


def hpo(default_parameters):
    global gp_seed, dde_seed, output_path
    dde.config.set_random_seed(dde_seed)

    search_result = gp_minimize(
        func=fitness,
        dimensions=dimensions,
        acq_func="PI",  # Probability Improvement.
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
    global ITERATION, gp_seed, dde_seed
    ITERATION = 0
    gp_seed = None
    dde_seed = None


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
            a['Metric']=a.apply(metric, axis=1)
            a["Metric"] = a["Metric"].apply(lambda x: '%.2e' % x)
            a["Config"] = a[["Learning Rate", "Num Dense Layers", "Num Dense Nodes",
                             "Activation"]].apply(lambda x: '[' + ','.join(map(str, x)) + ']', axis=1)
            b = a.loc[:, ["Config", "Metric", "Error"]].sort_values(by="Error", ascending=True)
            b["Iteration"] = b.index

            # Navigate back to the output folder
            os.chdir(output_fold)
            b.to_excel(writer, sheet_name=folder, index=False)


def plot_err(b):
    global output_path, msg
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
    plt.savefig(f"{output_path}figures/error_domain_{msg}.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{output_path}figures/error_time_{msg}.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{output_path}figures/error_space_{msg}.png", dpi=300, bbox_inches='tight')
    plt.show()


    # Save the data as an XLSX file
    data = {'Constant Flux': [np.linalg.norm(error_grid['c']), np.sum(np.linalg.norm(error_grid['c'], axis=0)), np.sum(np.linalg.norm(error_grid['c'], axis=1))],
            'Linear Flux': [np.linalg.norm(error_grid['l']), np.sum(np.linalg.norm(error_grid['l'], axis=0)), np.sum(np.linalg.norm(error_grid['l'], axis=1))],
            'Exponential Flux': [np.linalg.norm(error_grid['e']), np.sum(np.linalg.norm(error_grid['e'], axis=0)), np.sum(np.linalg.norm(error_grid['e'], axis=1))],
            'Sinusoidal Flux': [np.linalg.norm(error_grid['s']), np.sum(np.linalg.norm(error_grid['s'], axis=0)), np.sum(np.linalg.norm(error_grid['s'], axis=1))]}
    
    df = pd.DataFrame(data, index=['Domain', 'Space', 'Time'])
    df.to_excel(f"{output_path}/error.xlsx")




