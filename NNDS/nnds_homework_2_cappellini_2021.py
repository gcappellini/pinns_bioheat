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
class MeanAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="mean_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        matches = tf.equal(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(tf.cast(matches, tf.float32)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count


class BackwardTransfer(tf.keras.metrics.Metric):
    def __init__(self, name="backward_transfer", **kwargs):
        super(BackwardTransfer, self).__init__(name=name, **kwargs)
        self.initial_performance = self.add_weight(name="initial_perf", initializer="zeros")
        self.new_performance = self.add_weight(name="new_perf", initializer="zeros")

    def reset_states(self):
        self.initial_performance.assign(0.)
        self.new_performance.assign(0.)

    def update_initial_performance(self, value):
        self.initial_performance.assign(value)

    def update_new_performance(self, value):
        self.new_performance.assign(value)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return self.new_performance - self.initial_performance

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

# Head for Tissue Type (Classification, assuming 2 types for simplicity)
tissue_out = tf.keras.layers.Dense(2, activation='softmax', name='tissue_type')(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=[temp_out, perfusion_out, tissue_out])

# Print a summary of the model architecture
model.summary()


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
            if self.task_id in [0, 1]:  # regression tasks
                loss = tf.keras.losses.mean_squared_error(y, y_pred)
            else:  # classification task
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True)

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

        if self.task_id in [0, 1]:  # regression tasks
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        else:  # classification task
            mse_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        ewc_loss = 0.0
        for var in self.model.trainable_variables:
            if var.name in self.fisher:
                ewc_loss += tf.reduce_sum(self.fisher[var.name] * (var - self.star_vars[var.name]) ** 2)

        return mse_loss + self.lambda_ewc * ewc_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Task 1 (y1)
ewc_loss = EWCLoss(model, task_id=0)
model.compile(optimizer=optimizer, loss=ewc_loss, metrics=['mae'])
model.fit(X, y1, epochs=2, batch_size=32, validation_split=0.2)
ewc_loss.compute_fisher_information(X, y1)
ewc_loss.update_star_vars()

# Task 2 (y2)
ewc_loss = EWCLoss(model, task_id=1)
model.compile(optimizer=optimizer, loss=ewc_loss, metrics=['mae'])
model.fit(X, y2, epochs=2, batch_size=32, validation_split=0.2)
ewc_loss.compute_fisher_information(X, y2)
ewc_loss.update_star_vars()

from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"MeanAccuracy": MeanAccuracy})

# Task 3 (y3)
ewc_loss = EWCLoss(model, task_id=2)
model.compile(optimizer=optimizer, loss=ewc_loss, metrics=['MeanAccuracy'])
y3_squeezed = tf.squeeze(y3, axis=-1)
model.fit(X, y3_squeezed, epochs=4, batch_size=32, validation_split=0.2)

