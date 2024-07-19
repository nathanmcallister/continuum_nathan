#!/bin/python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from ANN import Model
from kinematics import tang_2_dcm, dcm_2_quat

model = Model(4, 6, [32, 32])

model.load("models/real_07_17_2024/2024_07_17_19_42_23.pt")

input = np.zeros(4)
input_tensor = torch.from_numpy(input)
print(input_tensor)

output_tensor = model(input_tensor)
print(output_tensor)
output = output_tensor.detach().numpy()
