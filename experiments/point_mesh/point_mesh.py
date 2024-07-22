#!/bin/python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from ANN import Model
from kinematics import tang_2_dcm, dcm_2_quat
import itertools
import pandas

model = Model(4, 6, [32, 32])

model.load("models/real_07_17_2024/2024_07_17_19_42_23.pt")

file = open(r"point_mesh.txt", "w")

mesh_range = np.linspace(-2, 2, 20)
positions = [mesh_range, mesh_range, mesh_range, mesh_range]
points = list(itertools.product(*positions))

for i in range(np.size(points, 0)):
    tensor = torch.tensor(points[i])
    output = model(tensor)
    output_pos = output[:3].detach().numpy()
    for j in range(np.size(output_pos)):
        if j < np.size(output_pos) - 1:
            file.write(str(output_pos[j]) + ",")
        else:
            file.write(str(output_pos[j]) + "\n")

# for i in range(np.size(points, 0)):
#     for j in range(np.size(points[i])):
#         if j < np.size(points[i]) - 1:
#             file.write(str(points[i][j]) + ",")
#         else:
#             file.write(str(points[i][j]) + "\n")

file.close()

# verify points via plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

file = open(r"point_mesh.txt", "r")

data = pandas.read_csv(file)

