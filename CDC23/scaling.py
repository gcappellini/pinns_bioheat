import numpy as np
import matplotlib.pyplot as plt
from mu import mu


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

dT = TM-Ta
q0_ad = q0/dT

W = np.linspace(W_min, W_max, 8)

print(q0_ad)
