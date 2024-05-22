#!/bin/python3
import time
import serial
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import kinematics
import utils_data
import continuum_arduino
import continuum_aurora

T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

wait_time = 2
ns2s = 10 ** (-9)
trajectory = np.loadtxt("output/trajectory.dat", delimiter=",")
cable_trajectory = np.loadtxt("output/cable_trajectory.dat", delimiter=",")
num_points = cable_trajectory.shape[1]

aurora = continuum_aurora.init_aurora()
probe_list = ["0A"]
arduino = continuum_arduino.init_arduino()

motor_setpoints = continuum_arduino.load_motor_setpoints("../../tools/motor_setpoints")
continuum_arduino.write_motor_vals(arduino, motor_setpoints)
time.sleep(2)

model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
model.load("../model_learning/models/real_05_12_2024a/2024_05_12_19_56_49.pt")
model.model.eval()

model_pos = np.nan * np.zeros((3, num_points))
model_tang = np.nan * np.zeros_like(model_pos)
true_pos = np.nan * np.zeros_like(model_pos)
true_tang = np.nan * np.zeros_like(model_pos)

for i in range(num_points):
    print(f"{i+1} of {num_points}")
    dls_np = cable_trajectory[:, i]
    dls_list = dls_np.tolist()
    motor_cmds = continuum_arduino.one_seg_dl_2_motor_vals(dls_list, motor_setpoints)
    continuum_arduino.write_motor_vals(arduino, motor_cmds)
    dls_tensor = torch.tensor(dls_np)
    output_tensor = model(dls_tensor)
    model_pos[:, i] = output_tensor[:3].detach().numpy()
    model_tang[:, i] = output_tensor[3:].detach().numpy()
    time.sleep(wait_time)
    trans = continuum_aurora.get_aurora_transforms(aurora, probe_list)
    T = continuum_aurora.get_T_tip_2_model(trans["0A"], T_aurora_2_model, T_tip_2_coil)
    true_pos[:, i] = T[0:3, 3]
    true_tang[:, i] = kinematics.dcm_2_tang(T[0:3, 0:3])

continuum_arduino.write_motor_vals(arduino, motor_setpoints)

arduino.close()
aurora.close()

np.savetxt("output/model_trajectory.dat", model_pos, delimiter=",")
np.savetxt("output/true_trajectory.dat", true_pos, delimiter=",")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(trajectory[0, :], trajectory[1, :], trajectory[2, :], label="Trajectory")
ax.scatter(model_pos[0, :], model_pos[1, :], model_pos[2, :], label="Model Prediction")
ax.scatter(true_pos[0, :], true_pos[1, :], true_pos[2, :], label="Truth Position")
plt.xlim((-32, 32))
plt.ylim((-32, 32))
ax.set_zlim((0, 64))
plt.title("Open Loop Trajectory Tracking")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.legend()
plt.show()
