#!/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import kinematics
import continuum_aurora

AURORA_CIRCLE_FILE = "../../data/base_positions/base_circle_04_22_24a.csv"
AURORA_TOP_FILE = "../../data/base_positions/base_top_04_22_24a.csv"

T_AURORA_2_MODEL_FILE = "../../tools/T_aurora_2_model"
T_TIP_2_COIL_FILE = "../../tools/T_tip_2_coil"

PENPROBE_FILE = "../../tools/penprobe7"

T_aurora_2_model = np.loadtxt(T_AURORA_2_MODEL_FILE, delimiter=",")
T_tip_2_coil = np.loadtxt(T_TIP_2_COIL_FILE, delimiter=",")

penprobe = np.loadtxt(PENPROBE_FILE, delimiter=",").transpose()

def base_circle() -> np.ndarray:
    circle_df = pd.read_table(AURORA_CIRCLE_FILE, sep=',', header=None)
    num_probes = len(pd.unique(circle_df[2]))

    penprobe_index = "0A"
    pen_table = circle_df[:][circle_df[2] == penprobe_index]

    pen_quat = pen_table.iloc[:, 3:7].to_numpy().transpose()
    pen_coil_pos = pen_table.iloc[:, 7:10].to_numpy().transpose()

    num_meas = pen_coil_pos.shape[1]

    tip_pos_in_aurora = np.ones((3, num_meas)) * np.inf

    for i in range(num_meas):
        dcm = kinematics.quat_2_dcm(pen_quat[:, i])
        tip_pos_in_aurora[:, i] = pen_coil_pos[:, i] + np.matmul(dcm, penprobe)

    assert not ((tip_pos_in_aurora == np.inf).any() or (tip_pos_in_aurora == np.nan).any())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(tip_pos_in_aurora[0,:], tip_pos_in_aurora[1, :], tip_pos_in_aurora[2, :])
    plt.show()

    tip_pos_in_model = kinematics.Tmult(T_aurora_2_model, tip_pos_in_aurora)
    
    radius = 18

    def minimization_func(x):
        delta = tip_pos_in_model[0:2, :] - x.reshape((2, 1))
        r = np.sqrt(np.sum(delta ** 2, axis=0))
        return np.sum((r - radius)**2)
    
    x0 = np.zeros(2)
    out = opt.minimize(minimization_func, x0, method="nelder-mead")
    center = out['x']

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(theta) + center[0]
    circle_y = radius * np.sin(theta) + center[1]

    plt.scatter(center[0], center[1])
    plt.plot(circle_x, circle_y)
    plt.scatter(tip_pos_in_model[0, :], tip_pos_in_model[1, :])
    ax = plt.gca()
    ax.set_aspect("equal")

    plt.show()

    return center


def base_top() -> float:
    top_df = pd.read_table(AURORA_TOP_FILE, sep=',', header=None)
    num_probes = len(pd.unique(top_df[2]))

    penprobe_index = "0A"
    pen_table = top_df[:][top_df[2] == penprobe_index]

    pen_quat = pen_table.iloc[:, 3:7].to_numpy().transpose()
    pen_coil_pos = pen_table.iloc[:, 7:10].to_numpy().transpose()

    num_meas = pen_coil_pos.shape[1]

    tip_pos_in_aurora = np.ones((3, num_meas)) * np.inf

    for i in range(num_meas):
        dcm = kinematics.quat_2_dcm(pen_quat[:, i])
        print(np.linalg.det(dcm))
        tip_pos_in_aurora[:, i] = pen_coil_pos[:, i] + np.matmul(dcm, penprobe)

    assert not ((tip_pos_in_aurora == np.inf).any() or (tip_pos_in_aurora == np.nan).any())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(tip_pos_in_aurora[0,:], tip_pos_in_aurora[1, :], tip_pos_in_aurora[2, :])
    plt.show()

    tip_pos_in_model = kinematics.Tmult(T_aurora_2_model, tip_pos_in_aurora)
    plt.figure(2)
    plt.plot(tip_pos_in_model[2,:])
    plt.show()
    
    return tip_pos_in_model[2, :].mean()

xy = base_circle()
z = base_top()

out = np.zeros(3)
out[0:2] = xy
out[2] = z

np.savetxt("../../tools/spine_pos", out, delimiter=",")
