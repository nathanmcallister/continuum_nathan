#!/bin/python3
import numpy as np
import utils_data

DATA_FILE = "cc_data.dat"

bingo = [np.array([1, 1]), np.array([2, 2])]
bongo = [np.array([3, 2, 1]), np.array([1, 2, 3])]

for bing, bong in zip(bingo, bongo):
    print("A,")
    for bing_chilling in bing: print(f"{bing_chilling},")
    for bong_chilling in bong: print(f"{bong_chilling},")


