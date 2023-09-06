import tensorflow as tf
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

# Define the custom SReLU activation layer
class SReLU(tf.keras.layers.Layer):
    def __init__(self, tr_init=1.0, ar_init=0.1, tl_init=-1.0, al_init=0.1):
        super(SReLU, self).__init__()
        self.tr = tf.Variable(tr_init, trainable=True, name='tr')
        self.ar = tf.Variable(ar_init, trainable=True, name='ar')
        self.tl = tf.Variable(tl_init, trainable=True, name='tl')
        self.al = tf.Variable(al_init, trainable=True, name='al')

    def call(self, inputs):
        s_greater_tr = tf.where(inputs > self.tr, self.tr + self.ar * (inputs - self.tr), inputs)
        s_between_tl_tr = tf.where(tf.logical_and(inputs > self.tl, inputs <= self.tr), inputs, s_greater_tr)
        s_less_tl = tf.where(inputs < self.tl, self.tl + self.al * (inputs - self.tl), s_between_tl_tr)
        return s_less_tl

# Define a custom metric for mean accuracy
class MeanAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='mean_accuracy', **kwargs):
        super(MeanAccuracy, self).__init__(name=name, **kwargs)
        self.mean_accuracy = self.add_weight(name='mean_accuracy', initializer='zeros')
        self.num_tasks = self.add_weight(name='num_tasks', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute accuracy for the current task
        accuracy = tf.keras.metrics.binary_accuracy(y_true, y_pred)

        # Update mean accuracy and number of tasks
        self.mean_accuracy.assign_add(tf.reduce_mean(accuracy))
        self.num_tasks.assign_add(1.0)

    def result(self):
        return self.mean_accuracy / self.num_tasks

# Define a custom metric for backward transfer
class BackwardTransfer(tf.keras.metrics.Metric):
    def __init__(self, name='backward_transfer', **kwargs):
        super(BackwardTransfer, self).__init__(name=name, **kwargs)
        self.backward_transfer = self.add_weight(name='backward_transfer', initializer='zeros')
        self.num_tasks = self.add_weight(name='num_tasks', initializer='zeros')

    def update_state(self, current_task_accuracy, previous_task_accuracies):
        # Compute backward transfer as the difference between current task accuracy
        # and the maximum accuracy achieved on previous tasks
        max_previous_accuracy = tf.reduce_max(previous_task_accuracies)
        bt = current_task_accuracy - max_previous_accuracy

        # Update backward transfer and number of tasks
        self.backward_transfer.assign_add(bt)
        self.num_tasks.assign_add(1.0)

    def result(self):
        return self.backward_transfer / self.num_tasks


# Calculate Fisher information matrix for a given task
def calculate_fisher_information(model, x, num_samples=100):
    fisher_information_matrices = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            num_params = layer.count_params()
            fisher_matrix = tf.zeros((num_params, num_params))

            for _ in range(num_samples):
                with tf.GradientTape() as tape:
                    sampled_x = x[np.random.choice(x.shape[0], size=num_samples)]
                    sampled_x = tf.convert_to_tensor(sampled_x, dtype=tf.float32)
                    predictions = layer(sampled_x)

                gradients = tape.gradient(predictions, layer.trainable_variables)
                flat_gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
                fisher_matrix += tf.outer(flat_gradients, flat_gradients)

            fisher_matrix /= num_samples
            fisher_information_matrices.append(fisher_matrix)

    return fisher_information_matrices

# Calculate EWC loss
def calculate_ewc_loss(model, fisher_information_matrices, previous_task_params, ewc_lambda=1.0):
    ewc_losses = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            num_params = layer.count_params()
            fisher_matrix = fisher_information_matrices[i]
            prev_params = previous_task_params[i]

            # Calculate EWC loss for the layer
            ewc_loss = 0.5 * ewc_lambda * tf.reduce_sum(tf.square(layer.trainable_variables[0] - prev_params[0])) + \
                       0.5 * ewc_lambda * tf.reduce_sum(tf.square(layer.trainable_variables[1] - prev_params[1]))

            ewc_losses.append(ewc_loss)

    return tf.reduce_sum(ewc_losses)

# Define a synthetic dataset and task splitting (customize this part for your dataset)
num_tasks = 3
task_data = OrderedDict()

# Split data into tasks
for task_id in range(num_tasks):
    task_data[task_id] = {
        'X': np.random.rand(100, 1),  # Input data
        'y': np.random.randint(2, size=(100, 1))  # Binary classification labels
    }

# Initialize EWC-related variables
fisher_information_matrices = {}  # Store Fisher information matrices for each task
previous_task_params = None  # Store parameters of previous tasks

# Create a multi-head model (customize this part based on your architecture)
def create_multi_head_model():
    input_layer = tf.keras.layers.Input(shape=(1,))
    heads = []

    for task_id in range(num_tasks):
        x = tf.keras.layers.Dense(32, activation=SReLU())(input_layer)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        head = tf.keras.models.Model(inputs=input_layer, outputs=output)
        heads.append(head)

    return heads

# Initialize the model
model_heads = create_multi_head_model()

# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Initialize custom metrics
mean_accuracy_metric = MeanAccuracy()
backward_transfer_metric = BackwardTransfer()

# Define the number of training epochs per task
epochs_per_task = 50

# Training loop
@tf.function  # Wrap the training loop in a tf.function for optimization
def train_task(task_id):
    current_task_data = task_data[task_id]
    X_task, y_task = current_task_data['X'], current_task_data['y']

    for epoch in range(epochs_per_task):
        with tf.GradientTape() as tape:
            y_pred = model_heads[task_id](X_task, training=True)
            loss = loss_fn(y_task, y_pred)

        gradients = tape.gradient(loss, model_heads[task_id].trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_heads[task_id].trainable_variables))

    # Calculate accuracy on the current task
    current_task_accuracy = mean_accuracy_metric(y_task, y_pred)

    # Calculate backward transfer
    if task_id > 0:
        previous_task_accuracies = []
        for prev_task_id in range(task_id):
            prev_task_data = task_data[prev_task_id]
            X_prev_task, y_prev_task = prev_task_data['X'], prev_task_data['y']
            y_pred_prev_task = model_heads[prev_task_id](X_prev_task, training=False)
            prev_task_accuracy = mean_accuracy_metric(y_prev_task, y_pred_prev_task)
            previous_task_accuracies.append(prev_task_accuracy)

        backward_transfer = backward_transfer_metric(current_task_accuracy, previous_task_accuracies)

    # Store EWC-related variables for future tasks
    if task_id > 0:
        fisher_information_matrices[task_id - 1] = calculate_fisher_information(model_heads[task_id - 1], X_task)
        previous_task_params = model_heads[task_id - 1].get_weights()

    # Apply EWC regularization to the model
    if task_id > 0:
        ewc_loss = calculate_ewc_loss(model_heads[task_id], fisher_information_matrices, previous_task_params)
        total_loss = loss + ewc_loss
    else:
        total_loss = loss

    # Update the model with EWC regularization
    gradients = tape.gradient(total_loss, model_heads[task_id].trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_heads[task_id].trainable_variables))

    # Display metrics
    print(f'Task {task_id + 1}: Accuracy={current_task_accuracy}, Backward Transfer={backward_transfer}')

# Run the training loop for each task
for task_id in range(num_tasks):
    train_task(task_id)

# Plot performance graphs with task IDs on the x-axis (customize as needed)
task_ids = list(range(1, num_tasks + 1))
mean_accuracies = [mean_accuracy_metric.result().numpy()] * num_tasks
backward_transfers = [backward_transfer_metric.result().numpy()] * num_tasks

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(task_ids, mean_accuracies, marker='o')
plt.xlabel('Task ID')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy per Task')

plt.subplot(1, 2, 2)
plt.plot(task_ids, backward_transfers, marker='o')
plt.xlabel('Task ID')
plt.ylabel('Backward Transfer')
plt.title('Backward Transfer per Task')

plt.tight_layout()
plt.show()
