import numpy as np
import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 1, steps=100)

# General parameters
L0 = 0.05
TM = 45
Ta = 37
tauf = 1800
qmet = 4200

# Fat tissue parameters
rho = 940
c = 2500
K = 0.2
k_eff = 5
alfa = rho * c / k_eff
# k_eff = k*(1+alfa*omegab)

W_avg = 0.54
W_min = 0.36
W_max = 0.72
cb = 3825

dT = TM - Ta

a1 = (alfa * (L0 ** 2)) / tauf
a2 = (L0 ** 2) * cb / k_eff
a3 = (L0 ** 2) / (k_eff * dT)

# Antenna parameters
beta = 1
p = 150/(1.75e-3)
cc = 16
X0 = 0.08



def source(s):
    return a3*(qmet + beta*torch.exp(-cc*L0*(X0-s))*p)

def perfusion(s):
    return torch.full_like(s, -a2*W_avg)

plotty = source(x) + perfusion(x)
plt.plot(x, plotty)
plt.show()
