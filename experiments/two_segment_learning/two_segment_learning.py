#!/bin/python3
import numpy as np

training_epochs = [1024, 2048, 4096]
model_sizes = [[32, 32], [64, 64], [32, 32, 32], [64, 64, 64], [128]]
datapoints = [2**13, 2**14, 2**15, 2**16]

test_size = 2**12

for epochs in traning_epochs:
    for size in model_sizes:
        for num_datapoints in datapoints:

