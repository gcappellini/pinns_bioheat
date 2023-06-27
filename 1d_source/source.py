import numpy as np
import torch
import matplotlib.pyplot as plt

# beta = 1
# p = 5
# c = 2
# z0 = 0.5
#
# def sar(s):
#     return beta*np.exp(-c*(z0-s))*p
#
#
x = np.linspace(0,1)



a1 = 1.061375
a2 = 1.9125
a3 = 6.25e-05
Q = 100
beta = 1
c = 16
L0 = 0.05
X0 = 0.08
p = 2000
W_avg = 2.3



def s(s):
    return a3*(Q+beta*np.exp(-c*L0*(X0-s))*p)

plt.plot(x, s(x))
plt.show()