import numpy as np
import matplotlib.pyplot as plt
import utils
import pandas as pd

# Inizio col caso 1
folder_path = ""
n, l = 3, "s"
def compare_networks(metrics_net1, metrics_net2, metrics_net3):
    metrics = ['MAE', 'Max Error', 'Standard Deviation']
    net1_values = [metrics_net1[metric] for metric in metrics]
    net2_values = [metrics_net2[metric] for metric in metrics]
    net3_values = [metrics_net3[metric] for metric in metrics]

    # Set the width of the bars
    bar_width = 0.25

    # Set the position of the bars on the x-axis
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the bar plot
    plt.bar(r1, net1_values, color='blue', width=bar_width, edgecolor='black', label='Network 1')
    plt.bar(r2, net2_values, color='orange', width=bar_width, edgecolor='black', label='Network 2')
    plt.bar(r3, net3_values, color='green', width=bar_width, edgecolor='black', label='Network 3')

    # Add xticks and labels
    plt.xticks([r + bar_width for r in range(len(metrics))], metrics)
    plt.ylabel('Error')
    plt.title('Comparison of Error Metrics between Networks')

    # Add a legend
    plt.legend()

    # Save and show the plot
    plt.savefig(f"{folder_path}compare.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_errors(error_values, ll, label):
    plt.plot(error_values)
    plt.xlabel('Time Step / Spatial Location')
    plt.ylabel('Absolute Error')
    plt.title(f'{label} Absolute Error')
    plt.savefig(f"{folder_path}{ll}_{label}.png", dpi=300, bbox_inches='tight')
    plt.show()


def calculate_accuracy_metrics(tt, predictions, ground_truth):
    metrics = {}

    # Calculate metrics...

    # Ensure predictions and ground_truth have the same shape
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - ground_truth))
    metrics['MAE'] = mae

    # Maximum Absolute Error
    max_error = np.max(np.abs(predictions - ground_truth))
    metrics['Max Error'] = max_error

    # Standard Deviation
    std_dev = np.std(np.abs(predictions - ground_truth))
    metrics['Standard Deviation'] = std_dev

    file_path = f"{folder_path}{tt}_errors.xlsx"

    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics, index=[0])

    # Save the DataFrame to an Excel file
    metrics_df.to_excel(file_path, index=False)

    error_grid = {}
    pred_grid = np.reshape(predictions, X.shape)
    truth_grid = np.reshape(ground_truth, X.shape)

    # Mean Absolute Error over Time (MAE_t)
    mae_t = np.mean(np.abs(pred_grid - truth_grid), axis=1)
    error_grid['MAE_t'] = mae_t.tolist()

    # Mean Absolute Error over Space (MAE_s)
    mae_s = np.mean(np.abs(pred_grid - truth_grid), axis=0)
    error_grid['MAE_s'] = mae_s.tolist()

    return metrics, error_grid

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


q0 = 16
dT = TM - Ta
q0_ad = q0/dT

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

ini1 = 101
ini2 = 706

confi1 = [0.014, 1, 75, "tanh", "Glorot normal"]
confi2 = [0.001255, 4, 45, "tanh", "He uniform"]

setts1 = [1, 1, 1, 1, 1, 1]
setts2 = [10, 10, 10, 1000, 100000, 1]

utils.inizia_esperimento(ini1, confi1, setts1)
a1 = utils.create_model(confi1, setts1)
p1 = utils.restore_model(a1, confi1, setts1)
predictions_net1 = p1.predict(XX[l])

utils.inizia_esperimento(ini2, confi2, setts1)
a2 = utils.create_model(confi2, setts1)
p2 = utils.restore_model(a2, confi2, setts1)
predictions_net2 = p2.predict(XX[l])

utils.inizia_esperimento(ini2, confi2, setts2)
a3 = utils.create_model(confi2, setts2)
p3 = utils.restore2_model(a3, confi2, setts2)
predictions_net3 = p3.predict(XX[l])

ground_truth = np.loadtxt(f"{folder_path}matlab/output_matlab_system_{n}.txt")[:, 2:]

# Calculate metrics for each neural network
metrics_net1, grid_net1 = calculate_accuracy_metrics("net1", predictions_net1, ground_truth)
metrics_net2, grid_net2 = calculate_accuracy_metrics("net2", predictions_net2, ground_truth)
metrics_net3, grid_net3 = calculate_accuracy_metrics("net3", predictions_net3, ground_truth)

# Print the metrics for each neural network
print("Metrics for Network 1:")
print(metrics_net1)
print()

print("Metrics for Network 2:")
print(metrics_net2)
print()

print("Metrics for Network 3:")
print(metrics_net3)
print()

# Plot the errors for each network
plot_errors(grid_net1["MAE_t"], "MAE_t", 'Network 1')
plot_errors(grid_net2["MAE_t"], "MAE_t", 'Network 2')
plot_errors(grid_net3["MAE_t"], "MAE_t", 'Network 3')

plot_errors(grid_net1["MAE_s"], "MAE_s", 'Network 1')
plot_errors(grid_net2["MAE_s"], "MAE_s", 'Network 2')
plot_errors(grid_net3["MAE_s"], "MAE_s", 'Network 3')

# Compare the error metrics between the networks
compare_networks(metrics_net1, metrics_net2, metrics_net3)
