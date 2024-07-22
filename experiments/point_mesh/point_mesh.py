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

mesh_range = np.linspace(-12, 12, 12)
positions = [mesh_range, mesh_range, mesh_range, mesh_range]
points = list(itertools.product(*positions))

for i in range(np.size(points, 0)):
    tensor = torch.tensor(points[i])
    output = model(tensor)
    output_pos = output[:3].detach().numpy()
    output_tang = output[3:6].detach().numpy()
    output_quat = dcm_2_quat(tang_2_dcm(output_tang))
    output_line = [*output_pos, *output_quat]
    for j in range(np.size(output_line)):
        if j < np.size(output_line) - 1:
            file.write(str(output_line[j]) + ",")
        else:
            file.write(str(output_line[j]) + "\n")

print("{} points written to point_mesh.txt".format(np.size(points, 0)))
file.close()

# verify points via plot

fig = plt.figure()
ax = fig.add_subplot()

file = open(r"point_mesh.txt", "r")
data = pandas.read_csv(file, header=None)
file.close()

print(data)

x = data.iloc[:,0]
y = data.iloc[:,1]
z = data.iloc[:,2]

# limiting
mask = (x>-1)*(x<1)
x = x[mask]; y = y[mask]; z = z[mask]

ax.scatter(y, z)
ax.axis('equal')
plt.title("mesh cross section (-1 < x < 1 mm)")
plt.xlabel("y position (mm)")
plt.ylabel("z position (mm)")
plt.show()