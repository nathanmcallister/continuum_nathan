#!/bin/python3
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
from kinematics import quat_2_dcm, tang_2_dcm
import camarillo_cc
import kinematics
import utils_cc
import utils_data

trajectory = np.loadtxt("output/nathan_trajectory.dat", delimiter=",")
trajectory_tensor = torch.tensor(trajectory)
num_points = trajectory.shape[1]
num_closed_loop_steps = 11
step_size = 1


model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)

model.load("../model_learning/models/real_07_17_2024/2024_07_17_19_42_23.pt")
model.model.eval()

array = list(np.float_([-12, -12, -12, -12]))

tensor = torch.tensor(array)
output = model(tensor)
# tensor([0, 0, 64, 0, 0, 0])
output_pos = output[:3]


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
    # x_hat = extend_z(model(dl_tensor)[0:3], tang_2_dcm(model(dl_tensor)[3:6]), 13) # added height of electrode array housing
    print("tip pos = {} \nelectrode pos = {}".format(model(dl_tensor)[0:3].detach().numpy(), x_hat.detach().numpy()))

    e = x_hat - x_star
    delta_dl = dl_tensor - dl_0

    loss = torch.dot(e, e) * weighting[0] + torch.dot(delta_dl, delta_dl) * weighting[1]

    loss.backward()
    grad = dl_tensor.grad.numpy()

    return (loss.item(), grad)

def extend_z(pos, rot, length):
    """
    Translate a transfrom in its Z direction.
    
    Input
    :param pos: 3 element position list (x,y,z)
    :param rot: 3x3 rotation matrix
    :param length: distance to translate
    
    Output
    :return: new 3 element position array
    """
    pos = pos.detach().numpy()

    z_dir = np.multiply(rot[2, :3], [-1, -1, 1])
    # print("z direction = {}".format(z_dir))
    new_pos = np.add(length * z_dir, pos)
    new_pos = torch.tensor(new_pos)

    return new_pos

open_loop_dls = np.zeros((4, num_points))

for i in range(num_points):
    print(f"{i+1} of {num_points}")
    dl_0 = 0
    if i == 0:
        dl_0 = torch.tensor(np.array([0.0] * 4))
    else:
        dl_0 = torch.tensor(open_loop_dls[:, i - 1].flatten())

    x_star = trajectory_tensor[:, i].flatten()

    optim_func = lambda x: loss_fcn(x, model, x_star, dl_0, (50.0, 1.0))

    result = opt.minimize(
        optim_func,
        dl_0.numpy(),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-12, 12), (-12, 12), (-12, 12), (-12, 12)],
    )
    open_loop_dls[:, i] = result["x"]

np.savetxt("output/nathan_cable_trajectory.dat", open_loop_dls, delimiter=",")
