#!/bin/python3
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import camarillo_cc
import kinematics
import utils_cc
import utils_data

trajectory = np.loadtxt("output/trajectory.dat", delimiter=",")
trajectory_tensor = torch.tensor(trajectory)
num_points = trajectory.shape[1]
num_closed_loop_steps = 11
step_size = 1


model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
model.load("../model_learning/models/real_05_12_2024a/2024_05_12_19_56_49.pt")
model.model.eval()


def loss_fcn(
    dl: np.ndarray,
    model: ANN.Model,
    x_star: torch.tensor,
    dl_0: torch.tensor,
    weighting: Tuple[float, float],
) -> Tuple[float, np.ndarray]:

    dl_tensor = torch.tensor(dl, requires_grad=True)
    model.zero_grad()
    x_hat = model(dl_tensor)[0:3]

    e = x_hat - x_star
    delta_dl = dl_tensor - dl_0

    loss = torch.dot(e, e) * weighting[0] + torch.dot(delta_dl, delta_dl) * weighting[1]

    loss.backward()
    grad = dl_tensor.grad.numpy()

    return (loss.item(), grad)


open_loop_dls = np.zeros((4, num_points))

for i in range(num_points):
    dl_0 = 0
    if i == 0:
        dl_0 = torch.tensor(np.array([0.0] * 4))
    else:
        dl_0 = torch.tensor(open_loop_dls[:, i - 1].flatten())

    x_star = trajectory_tensor[:, i].flatten()

    optim_func = lambda x: loss_fcn(x, model, x_star, dl_0, (5.0, 1.0))

    result = opt.minimize(optim_func, dl_0.numpy(), method="BFGS", jac=True)
    open_loop_dls[:, i] = result["x"]

np.savetxt("output/cable_trajectory.dat", open_loop_dls, delimiter=",")
