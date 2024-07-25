#!/bin/python3
import numpy as np
from ANN import PoseLoss
from multi_input import MultiInputModel, MultiInputDataset

model = MultiInputModel(4, 1, 1, [32, 32], loss=PoseLoss())
dataset = MultiInputDataset(1, "./training_data/kinematic_2024_07_16_21_33_12.dat")

print(model)
print(dataset[0], dataset[1])

print(len(dataset))
dataset.clean()
print(len(dataset))
