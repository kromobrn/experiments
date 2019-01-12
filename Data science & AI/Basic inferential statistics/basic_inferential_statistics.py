import numpy as np
import pandas as pd
import scipy.stats as st

print(40 + 1.96 * (2.71))
print(40 - 1.96 * (2.71))

print(1.96 * (14.9/(32 ** .5)))
print(1.96 * (14.9/(32 ** .5)))

print(40 + 2.33 * (16.04/(250 ** .5)))
print(40 - 2.33 * (16.04/(250 ** .5)))

print(0.13 - (1.96))

engagement = pd.read_csv('D:\Source\experiments\Data science & AI\Basic inferential statistics\engagement_ratio_data.csv', header=None)
engagement = engagement.iloc[:,0]

engagement.std()
engagement.describe()

sample = np.array([8, 9, 12, 13, 14, 16])
pop_std = 2.8
sample.mean() # 12

print(12 - 1.96 * (2.8/(6 ** .5)))
print(12 + 1.96 * (2.8/(6 ** .5)))

print(st.norm.ppf(.999))

print(2.41/30**.5) # 0.44000378786248345
print((8.3-7.47)/0.44000378786248345)

print(2.41/50**.5) # 0.3408254685319159
print((8.3-7.47)/0.3408254685319159)

print(230/5**.5) # 102.85912696499032
print((9640-7895)/102.85912696499032) # 16.964950524944058