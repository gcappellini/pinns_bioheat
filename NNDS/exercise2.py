import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

# Create a synthetic dataset
np.random.seed(0)
x = np.linspace(0, 1, 1000)
t = np.random.rand(1000)
X = np.vstack((x, t)).T

a = [0.0008729, 3, 25, "silu", "He normal", 1, 1, 1, 1]
W_ref = 2.3

b = utils.create_system(a, W_ref)
# ss = utils.train_model(b, "sys")
ss = utils.restore_model(b, "sys")

y = ss.predict(X)

# Define the custom SReLU activation component
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

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=SReLU(), input_shape=(2,)),
    tf.keras.layers.Dense(32, activation=SReLU()),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the dataset
history = model.fit(X, y, epochs=100, verbose=1)

# Plot the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
