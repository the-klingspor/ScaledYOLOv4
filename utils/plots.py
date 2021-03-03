import numpy as np
import matplotlib.pyplot as plt

from active_learning.scoring import *


if __name__ == '__main__':
    x = np.linspace(0, 1, 200)
    conf_thres = 0.05

    x_lin = v_scale_lin(x, conf_thres=conf_thres, target=0.5)

    ent = entropy_bern(x)
    ent_lin = entropy_bern(x_lin)

    fig_lin = plt.figure()
    ax = fig_lin.add_subplot()
    ax.grid(True, which='both')
    ax.set_aspect('equal')
    plt.plot(x, x_lin, 'red', label="linear scaled input")
    plt.plot(x, ent, 'g', label="entropy")
    plt.plot(x, ent_lin, 'b', label="scaled entropy")

    plt.legend(loc='best')

    plt.savefig("lin_ent")
    plt.show()

    # Sigmoid scaling and entropy
    x_sig = v_scale_sigmoid(x, conf_thres=conf_thres, k=1e-2)
    ent_sig = entropy_bern(x_sig)

    fig_sig = plt.figure()
    ax_2 = fig_sig.add_subplot()
    ax_2.grid(True, which='both')
    ax_2.set_aspect('equal')
    plt.plot(x, x_sig, 'r', label="sigmoid scaled input")
    plt.plot(x, ent, 'g', label="entropy")
    plt.plot(x, ent_sig, 'b', label="scaled entropy")

    plt.legend(loc='best')

    plt.savefig("sig_ent")
    plt.show()

    # Minkowski scaling and entropy
    conf_thres = 0.95
    x_mink = v_scale_mink(x, conf_thres)
    ent_mink = entropy_bern(x_mink)

    fig_mink = plt.figure()
    ax_3 = fig_mink.add_subplot()
    ax_3.grid(True, which='both')
    ax_3.set_aspect('equal')
    plt.plot(x, x_mink, 'r', label="Minkowski scaled input")
    plt.plot(x, ent, 'g', label="entropy")
    plt.plot(x, ent_mink, 'b', label="scaled entropy")

    plt.legend(loc='best')

    name = "mink_ent" + str(int(conf_thres * 100))
    plt.savefig(name)
    plt.show()
