import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(f"matlab/matlab_pde.txt")
x, t, theta_sup, exact, perf, tissue = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:3].T, data[:, 3:4].T, data[:, 4:5].T, data[:, 5:].T
X = np.vstack((x, t, theta_sup)).T
y1 = exact.flatten()[:, None]
y2 = perf.flatten()[:, None]
y3 = tissue.flatten()[:, None]

class SReLU(tf.keras.layers.Layer):
    def __init__(self, tr_init=0.4, ar_init=2.0, tl_init=-0.4, al_init=0.4):
        super(SReLU, self).__init__()
        self.tr = tf.Variable(tr_init, trainable=True, name='tr')
        self.ar = tf.Variable(ar_init, trainable=True, name='ar')
        self.tl = tf.Variable(tl_init, trainable=True, name='tl')
        self.al = tf.Variable(al_init, trainable=True, name='al')

    def call(self, inputs):
        s_greater_tr = tf.where(inputs > self.tr, self.tr + self.ar * (inputs - self.tr), inputs)
        s_between_tl_tr = tf.where(tf.logical_and(inputs > self.tl, inputs < self.tr), inputs, s_greater_tr)
        s_less_tl = tf.where(inputs <= self.tl, self.tl + self.al * (inputs - self.tl), s_between_tl_tr)
        return s_less_tl

    def compute_output_shape(self, input_shape):
        return input_shape

# Create an instance of the SReLU layer
srelu_layer = SReLU()

# Create a range of values for plotting
x = np.linspace(-1, 1, 400)
x_tensor = tf.constant(x, dtype=tf.float32)

# Calculate the SReLU activation and its derivative
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x_tensor)
    srelu_activation = srelu_layer(x_tensor)

srelu_derivative = tape.gradient(srelu_activation, x_tensor)

# Plot the SReLU activation function and its derivative
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, srelu_activation, label='SReLU Activation')
plt.title('SReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')

plt.subplot(1, 2, 2)
plt.plot(x, srelu_derivative, label='SReLU Derivative')
plt.title('SReLU Derivative')
plt.xlabel('Input')
plt.ylabel('Derivative')
plt.tight_layout()
plt.savefig(f"SReLU.png", dpi=300, bbox_inches='tight')
plt.show()


# Exercise 3 (6 points): Continual learning


# Define metrics here:
performances = {i: [] for i in range(3)}  #


# Define the multi-head model
# Tip: the functional API is pretty good here

# Shared Base
input_layer = tf.keras.layers.Input(shape=(3,))
x = tf.keras.layers.Dense(128, activation=SReLU())(input_layer)
x = tf.keras.layers.Dense(128, activation=SReLU())(x)

# Head for Temperature (Regression)
temp_out = tf.keras.layers.Dense(1, name='temperature')(x)

# Head for Perfusion (Regression)
perfusion_out = tf.keras.layers.Dense(1, name='perfusion')(x)

# Head for Tissue Type (Regression)
tissue_out = tf.keras.layers.Dense(1, name='tissue_type')(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=[temp_out, perfusion_out, tissue_out])

# Print a summary of the model architecture
model.summary()

def evaluate_on_task(task_id, X, y):
    if task_id == 0:
        y_pred = model.predict(X)[0]
    elif task_id == 1:
        y_pred = model.predict(X)[1]
    else:  # task_id == 2
        y_pred = model.predict(X)[2]

    if task_id in [0, 1, 2]:  # All tasks are regression for now
        mae = tf.keras.losses.MAE(y, y_pred).numpy().mean()
        return mae
    return None


class EWCLoss(tf.keras.losses.Loss):
    def __init__(self, model, task_id, lambda_ewc=1e4):
        super().__init__()
        self.model = model
        self.task_id = task_id
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.star_vars = {}

    def compute_fisher_information(self, x, y):
        with tf.GradientTape() as tape:
            y_preds = self.model(x)
            y_pred = y_preds[self.task_id]
            if self.task_id in [0, 1, 2]:  # regression tasks
                loss = tf.keras.losses.mean_squared_error(y, y_pred)
            else:  # classification task
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=False)

        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Compute Fisher information
        for var, grad in zip(self.model.trainable_variables, grads):
            if grad is not None:
                if var.name in self.fisher:
                    self.fisher[var.name] += grad ** 2 / len(x)
                else:
                    self.fisher[var.name] = grad ** 2 / len(x)

    def update_star_vars(self):
        for var in self.model.trainable_variables:
            self.star_vars[var.name] = var.numpy()

    def call(self, y_true, y_pred_list):
        y_pred = y_pred_list[self.task_id]

        if self.task_id in [0, 1, 2]:  # regression tasks
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        else:  # classification task
            mse_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        ewc_loss = 0.0
        for var in self.model.trainable_variables:
            if var.name in self.fisher:
                ewc_loss += tf.reduce_sum(self.fisher[var.name] * (var - self.star_vars[var.name]) ** 2)

        return mse_loss + self.lambda_ewc * ewc_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
dummy_y = np.zeros_like(y1)

# Task 1 (y1)
ewc_loss = EWCLoss(model, task_id=0)
model.compile(optimizer=optimizer, loss=[ewc_loss, 'mse', 'mse'], metrics=['mae', None, None])
model.fit(X, [y1, dummy_y, dummy_y], epochs=2, batch_size=32, validation_split=0.2)
ewc_loss.compute_fisher_information(X, y1)
ewc_loss.update_star_vars()
performances[0].append(evaluate_on_task(0, X, y1))

# Task 2 (y2)
ewc_loss = EWCLoss(model, task_id=1)
model.compile(optimizer=optimizer, loss=[ewc_loss, 'mse', 'mse'], metrics=[None, 'mae', None])
model.fit(X, [y1, y2, dummy_y], epochs=2, batch_size=32, validation_split=0.2)
ewc_loss.compute_fisher_information(X, y2)
ewc_loss.update_star_vars()
performances[0].append(evaluate_on_task(0, X, y1))
performances[1].append(evaluate_on_task(1, X, y2))

# Task 3 (y3)
ewc_loss = EWCLoss(model, task_id=2)
model.compile(optimizer=optimizer, loss=[ewc_loss, 'mse', 'mse'], metrics=[None, None, 'mae'])
model.fit(X, [y1, y2, y3], epochs=2, batch_size=32, validation_split=0.2)
performances[0].append(evaluate_on_task(0, X, y1))
performances[1].append(evaluate_on_task(1, X, y2))
performances[2].append(evaluate_on_task(2, X, y3))


tasks = ["Task 1", "Task 2", "Task 3"]

# x-coordinates
x1 = ["Task 1", "Task 2", "Task 3"]
x2 = ["Task 2", "Task 3"]
x3 = ["Task 3"]

# Plotting
plt.plot(x1, performances[0], '-o', label="Performance on Task 1")
plt.plot(x2, performances[1], '-o', label="Performance on Task 2")
plt.plot(x3, performances[2], '-o', label="Performance on Task 3")

plt.xticks(ticks=[0, 1, 2], labels=tasks)
plt.ylabel("Performance (MAE)")
plt.xlabel("Tasks Trained On")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"MAE.png", dpi=300, bbox_inches='tight')
plt.show()
#
#
# # Calculate backward transfer for each task
# backward_transfers = {}
# for i in range(1, 3):  # We start from task 2 since task 1 has no backward transfer
#     initial_performance = performances[i][0]
#     final_performance = performances[i][-1]
#     backward_transfer = final_performance - initial_performance
#     backward_transfers[tasks[i]] = backward_transfer
#
# print("Backward Transfers:", backward_transfers)

r = np.zeros((3, 3))

# Populate the matrix
r[0, :] = performances[0]
r[1, 1:] = performances[1]
r[2, 2] = performances[2][0]

print("R matrix:", r)

den = 0.5*len(tasks)*(len(tasks)+1)
global_acc = r.sum()/den
backward_transf = (r[0, 1]-r[0, 0]+r[0, 2]-r[0, 0]+r[0, 2]-r[0, 1]+r[1, 2] - r[1, 1])/den

print("Accuracy:", global_acc)
print("Global backward transfer", backward_transf)
