#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(size=(1000,))
x2 = x ** 2
plt.hist(x2, 100)
plt.show()
