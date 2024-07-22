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

mesh_range = np.linspace(-2, 2, 10)
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

print("{} points written to point_mesh.txt".format(np.size(points, 0)))
file.close()

# # verify points via plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

file = open(r"point_mesh.txt", "r")
data = pandas.read_csv(file, header=None)
file.close()

print(data)

x = data.iloc[:, 0]
y = data.iloc[:, 1]
z = data.iloc[:, 2]

ax.scatter(x, y, z)
plt.show()


