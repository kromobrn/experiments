import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

val = np.random.normal(50, size=100)
#val = np.random.randint(0, 100, size=100)

lags = 50

print(f'values: {val}')
r = {}

for i in range(0, lags):
    
    lag = np.roll(val, shift=i)
    r[i] = st.pearsonr(val, lag)[0]

    print(f'{i}-th lag: {lag}', f'r={r[i]}')

plt.plot(list(r.values()))
#plt.xticks(list(r.keys()))
plt.title('r(values, i-th lag)')
plt.xlabel('lag')

plt.show()