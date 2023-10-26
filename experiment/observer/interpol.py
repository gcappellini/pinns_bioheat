import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import utils

current_file = os.path.abspath(__file__)
folder_pa = os.path.dirname(current_file)
folder_path = f"{folder_pa}/"
# output_path = None
output_path = folder_path

xold = np.linspace(0, 10, num=11, endpoint=True)
# y = np.cos(-x**2/9.0)
XO, y_true = utils.gen_obsdata()
instants = np.unique(XO[:, 4:5])
tot = np.hstack((XO, y_true))
XO_all = tot[tot[:, 0]==np.max(tot[:, 0])]

y = XO_all[:, 5:].reshape(len(instants),)

f = interp1d(instants, y)
f2 = interp1d(instants, y, kind='cubic')

"DEVO INTERPOLARE ANCHE X, TEND, TSUP, FL PER L'OSSERVATORE!!!!"

xnew = np.linspace(0, 0.1, num=41, endpoint=True)

plt.plot(instants, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.savefig(f"{folder_path}figures/prova_interp_new.png", dpi=300, bbox_inches='tight')
plt.show()



