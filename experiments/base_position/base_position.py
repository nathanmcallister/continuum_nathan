#!/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import utils_data
import kinematics
import continuum_aurora

PENPROBE_FILE = "../../tools/penprobe7"
REG_FILE = "../../data/regs/reg_04_25_24b.csv"
CIRCLE_FILE = "../../data/base_positions/base_circle_04_25_24a.csv"
TOP_FILE = "../../data/base_positions/base_top_04_25_24a.csv"

T_SW_2_MODEL_FILE = "../../tools/T_sw_2_model"
MODEL_TRUTH_FILE = "../../tools/12_model_registration_points_in_sw"

num_repetitions = 5

T_sw_2_model = np.loadtxt(T_SW_2_MODEL_FILE, delimiter=",")
model_truth_in_sw = np.loadtxt(MODEL_TRUTH_FILE, delimiter=",")

model_truth_in_model = kinematics.Tmult(T_sw_2_model, model_truth_in_sw)

temp_model = np.zeros((3, model_truth_in_model.shape[1] * num_repetitions))

for i in range(num_repetitions):
    temp_model[:, i::num_repetitions] = model_truth_in_model


model_truth_in_model = temp_model

penprobe = np.loadtxt(PENPROBE_FILE, delimiter=",")

reg_transforms = utils_data.parse_aurora_csv(REG_FILE)
circle_transforms = utils_data.parse_aurora_csv(CIRCLE_FILE)
top_transforms = utils_data.parse_aurora_csv(TOP_FILE)

reg_pos = kinematics.penprobe_transform(penprobe, reg_transforms['0A'])
circle_pos = kinematics.penprobe_transform(penprobe, circle_transforms['0A'])
top_pos = kinematics.penprobe_transform(penprobe, top_transforms['0A'])

reg_pos_in_model, T_aurora_2_model, rmse = kinematics.rigid_align_svd(reg_pos, model_truth_in_model)

circle_pos_in_model = kinematics.Tmult(T_aurora_2_model, circle_pos)
top_pos_in_model = kinematics.Tmult(T_aurora_2_model, top_pos)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(reg_pos_in_model[0, :], reg_pos_in_model[1, :], reg_pos_in_model[2, :])
ax.scatter(circle_pos_in_model[0, :], circle_pos_in_model[1, :], circle_pos_in_model[2, :])
ax.scatter(top_pos_in_model[0, :], top_pos_in_model[1, :], top_pos_in_model[2, :])

plt.show()

def base_circle() -> np.ndarray:
    radius = 18

    def minimization_func(x):
        delta = circle_pos_in_model[0:2, :] - x.reshape((2, 1))
        r = np.sqrt(np.sum(delta ** 2, axis=0))
        return np.sum((r - radius)**2)
    
    x0 = np.zeros(2)
    out = opt.minimize(minimization_func, x0, method="nelder-mead")
    center = out['x']

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(theta) + center[0]
    circle_y = radius * np.sin(theta) + center[1]

    plt.scatter(center[0], center[1], marker='+')
    plt.plot(circle_x, circle_y)
    plt.scatter(circle_pos_in_model[0, :], circle_pos_in_model[1, :])
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.title("Measured Horizontal Position of Base in Model Frame")
    plt.legend([f"Circle Center ({center[0]:.2f}, {center[1]:.2f})", "Model Base", "Measurements"], loc="upper left")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    plt.show()

    return center


def base_top() -> float:
    mean_val = top_pos_in_model[2, :].mean()
    plt.figure(2)
    plt.plot(top_pos_in_model[2,:], 'o')
    plt.plot([0, top_pos_in_model.shape[1]-1], [mean_val, mean_val])
    plt.title("Measured Height of Base in Model Frame")
    plt.xlabel("Measurement")
    plt.ylabel("z (mm)")
    plt.legend(["Measurements", f"Average Height {mean_val:.3f} mm"])
    plt.show()

    return mean_val

base_circle()
base_top()
