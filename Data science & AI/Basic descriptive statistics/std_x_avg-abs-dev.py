import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

samples = []

for i in range(100):
    samples.append(np.random.rand(10) * 100)

std_devs = [array.std(ddof=1) for array in samples]
avg_abs_devs = [np.abs([xi - array.mean() for xi in array]).mean() for array in samples]

plt.scatter(std_devs, avg_abs_devs)
plt.show()