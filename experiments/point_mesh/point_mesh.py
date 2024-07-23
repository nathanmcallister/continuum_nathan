#!/bin/python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from ANN import Model
from kinematics import tang_2_dcm, dcm_2_quat
import itertools
import pandas

''' 
created 7/22/25 by Nathan McAllister

    Generates mesh of points from a model and writes them to a .txt
    .txt file can be brought into solidworks via Scan to 3D feature to create a surface
    this is useful for designing an accurate phantom or any surface that the arm can trace
    
    '''

def generate_mesh(model, lengths, file):

    # for each set of cable lengths
    for i in range(np.size(lengths, 0)):

        # get position & tang from forward model
        tensor = torch.tensor(lengths[i])
        output = model(tensor)
        output_pos = output[:3].detach().numpy()
        output_tang = output[3:6].detach().numpy()
        output_quat = dcm_2_quat(tang_2_dcm(output_tang))
        output_line = [*output_pos, *output_quat]

        # then write to file
        for j in range(np.size(output_line)):
            if j < np.size(output_line) - 1:
                file.write(str(output_line[j]) + ",")
            else:
                file.write(str(output_line[j]) + "\n")


def plot_cross_section(filename):

    # create fig
    fig = plt.figure()
    ax = fig.add_subplot()

    # open, read, and close file
    file = open(filename, "r")
    data = pandas.read_csv(file, header=None)
    file.close()

    # seperate into x, y, z lists
    x = data.iloc[:,0]; y = data.iloc[:,1]; z = data.iloc[:,2]

    # selecting specific range of x values
    mask = (x>-1)*(x<1)
    x = x[mask]; y = y[mask]; z = z[mask]

    # plot
    ax.scatter(y, z)
    ax.axis('equal')
    plt.title("mesh cross section (-1 < x < 1 mm)")
    plt.xlabel("y position (mm)")
    plt.ylabel("z position (mm)")
    plt.show()


model = Model(4, 6, [32, 32])
model.load("models/real_07_17_2024/2024_07_17_19_42_23.pt")

mesh_range = np.linspace(-12, 0, 20)
positions = [mesh_range, mesh_range, mesh_range, mesh_range]
lengths = list(itertools.product(*positions))
filename = r"point_mesh.txt"

file = open(filename, "w")
loose = [12]

range_q1 = [mesh_range, mesh_range, loose, loose]
lengths_q1 = list(itertools.product(*range_q1))
generate_mesh(model, lengths_q1, file)

range_q2 = [loose, mesh_range, mesh_range, loose]
lengths_q2 = list(itertools.product(*range_q2))
generate_mesh(model, lengths_q2, file)

range_q3 = [loose, loose, mesh_range, mesh_range]
lengths_q3 = list(itertools.product(*range_q3))
generate_mesh(model, lengths_q3, file)

range_q4 = [mesh_range, loose, loose, mesh_range]
lengths_q4 = list(itertools.product(*range_q4))
generate_mesh(model, lengths_q4, file)

print("{} points written to point_mesh.txt".format(4 * np.size(lengths_q1, 0)))
file.close()

file = open(filename, "r")
plot_cross_section(filename)
file.close()