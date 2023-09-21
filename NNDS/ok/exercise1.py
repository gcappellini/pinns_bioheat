import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"

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
plt.savefig(f"{folder_path}figures/SReLU.png", dpi=300, bbox_inches='tight')
plt.show()
