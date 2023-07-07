import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mu import mu

ll = [5, 200]

for lam in ll:
    o1, o2, o3, o4, o5, o6, o7, o8 = 0, 1, 2, 3, 4, 5, 6, 7


    def f(t, p):  # beware of argument order

        p1, p2, p3, p4, p5, p6, p7, p8 = p
        e1 = np.exp(-1 * mu(o1, t))
        e2 = np.exp(-1 * mu(o2, t))
        e3 = np.exp(-1 * mu(o3, t))
        e4 = np.exp(-1 * mu(o4, t))
        e5 = np.exp(-1 * mu(o5, t))
        e6 = np.exp(-1 * mu(o6, t))
        e7 = np.exp(-1 * mu(o7, t))
        e8 = np.exp(-1 * mu(o8, t))
        d = p1 * e1 + p2 * e2 + p3 * e3 + p4 * e4 + p5 * e5 + p6 * e6 + p7 * e7 + p8 * e8
        fp1 = - lam * (1 - (e1 / d)) * p1
        fp2 = - lam * (1 - (e2 / d)) * p2
        fp3 = - lam * (1 - (e3 / d)) * p3
        fp4 = - lam * (1 - (e4 / d)) * p4
        fp5 = - lam * (1 - (e5 / d)) * p5
        fp6 = - lam * (1 - (e6 / d)) * p6
        fp7 = - lam * (1 - (e7 / d)) * p7
        fp8 = - lam * (1 - (e8 / d)) * p8
        return fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8


    sol = integrate.solve_ivp(f, (0, 1), (1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8),
                              t_eval=np.linspace(0, 1, 100))
    x1, x2, x3, x4, x5, x6, x7, x8 = sol.y
    t = sol.t
    weights = np.zeros((sol.y.shape[0]+1, sol.y.shape[1]))
    weights[0] = sol.t
    weights[1:] = sol.y
    np.save(f'new_weights_lambda_{lam}.npy', weights)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(1800 * t, x1, alpha=1.0, linewidth=1.8, color='C3', label="Weight $p_1$")
    plt.plot(1800 * t, x2, alpha=1.0, linewidth=1.8, color='lime', label="Weight $p_2$")
    plt.plot(1800 * t, x3, alpha=1.0, linewidth=1.8, color='blue', label="Weight $p_3$")
    plt.plot(1800 * t, x4, alpha=1.0, linewidth=1.8, color='purple', label="Weight $p_4$")
    plt.plot(1800 * t, x5, alpha=1.0, linewidth=1.8, color='aqua', label="Weight $p_5$")
    plt.plot(1800 * t, x6, alpha=1.0, linestyle="dashed", linewidth=1.8, color='lightskyblue', label="Weight $p_6$")
    plt.plot(1800 * t, x7, alpha=1.0, linestyle="dashed", linewidth=1.8, color='darkred', label="Weight $p_7$")
    plt.plot(1800 * t, x8, alpha=1.0, linewidth=1.8, color='k', label="Weight $p_8$")

    ax1.set_xlim(0, 1800)
    ax1.set_ylim(bottom=0.0)

    # plt.xticks(np.arange(0, 200, 1801))
    ax1.set_xlabel(xlabel=r"Time t")  # xlabel
    ax1.set_ylabel(ylabel=r"Weights $p_j$")  # ylabel
    ax1.legend()
    ax1.set_title(f"Dynamic weights, $\lambda={lam}$", weight='semibold')
    plt.grid()
    plt.savefig(f"figures/mu_real/p_lam_{lam}.png",
                dpi=150, bbox_inches='tight')

    plt.show()


