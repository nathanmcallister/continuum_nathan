#!/bin/python3
import time
import serial
import numpy as np
import scipy.optimize as opt
import torch
import glob
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import kinematics
import utils_data
import continuum_arduino
import continuum_aurora


def calc_jacobian(u0, model):
    u0_tensor = torch.tensor(u0.flatten(), requires_grad=True)
    model.model.zero_grad()
    model_out = model(u0_tensor)
    unit_vectors = [
        torch.tensor(x)
        for x in [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ]
    ]

    jac = torch.stack(
        [
            torch.autograd.grad(model(u0_tensor), u0_tensor, vec)[0]
            for vec in unit_vectors
        ]
    ).numpy()

    return jac


T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

wait_time = 0.2
closed_loop_steps = 10
step_size = 0.01
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

model_filenames = glob.glob("../model_learning/models/real_05_12_2024a/*.pt")

models = []
for model_file in model_filenames:
    model = ANN.Model(
        input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
    )
    model.load(model_file)
    model.model.eval()
    models.append(model)

model_pos = np.nan * np.zeros((3, num_points))
model_tang = np.nan * np.zeros_like(model_pos)
true_pos = np.nan * np.zeros((3, closed_loop_steps * num_points))
true_tang = np.nan * np.zeros_like(true_pos)

for i in range(num_points):
    print(f"Trajectory point {i+1} of {num_points}")

    dls_np = cable_trajectory[:4, i]
    model_idx = int(cable_trajectory[4, i])

    deltas = np.zeros_like(dls_np)
    jac = calc_jacobian(dls_np, models[model_idx])

    dls_tensor = torch.tensor(dls_np, requires_grad=True)
    output_tensor = models[model_idx](dls_tensor)
    model_pos[:, i] = output_tensor[:3].detach().numpy()
    model_tang[:, i] = output_tensor[3:].detach().numpy()

    for j in range(closed_loop_steps):
        print(f"Closed loop step {j+1} of {closed_loop_steps}")
        dls_list = (dls_np + deltas).tolist()
        motor_cmds = continuum_arduino.one_seg_dl_2_motor_vals(
            dls_list, motor_setpoints
        )
        continuum_arduino.write_motor_vals(arduino, motor_cmds)

        time.sleep(wait_time)
        trans = continuum_aurora.get_aurora_transforms(aurora, probe_list)
        T = continuum_aurora.get_T_tip_2_model(
            trans["0A"], T_aurora_2_model, T_tip_2_coil
        )
        true_pos[:, closed_loop_steps * i + j] = T[0:3, 3]
        true_tang[:, closed_loop_steps * i + j] = kinematics.dcm_2_tang(T[0:3, 0:3])

        error = true_pos[:, closed_loop_steps * i + j] - trajectory[:, i]

        deltas -= np.transpose(jac) @ error * step_size

continuum_arduino.write_motor_vals(arduino, motor_setpoints)

arduino.close()
aurora.close()

np.savetxt("output/model_trajectory.dat", model_pos, delimiter=",")
np.savetxt("output/closed_loop_trajectory.dat", true_pos, delimiter=",")
