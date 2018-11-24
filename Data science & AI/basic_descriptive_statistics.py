import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

# sample = pd.Series([18, 20, 21, 18, 23, 15, 17, 22, 21])
# print sample.std(ddof=1)

sheet = pd.read_csv("Untitled spreadsheet - Sheet1.csv")

facebook_friends = sheet.iloc[:, [0,1]]
facebook_friends['Friends'] = facebook_friends.iloc[:, [1]].values

facebook_friends = facebook_friends.drop(facebook_friends.columns[1], axis='columns')

f = facebook_friends['Friends']
mean = f.mean() # 584.7407407407408
dev = (f - mean)
avg_dev = dev.mean() # -1.0105496686365869e-13
ss = (dev ** 2).sum() # 4978411.185185186
var = (dev ** 2).mean() # 184385.59945130316
std = f.std(ddof=0) # 429.40144323383817
std = f.std(ddof=1) # 437.5812533419974

mean + std # 1014.142183974579
mean - std # 155.3392975069026
f = f.dropna()
len(f) # 27
len(f.values[(f.values > mean - std) & (f.values < mean + std)]) # 18

18/27.0 # 0.6666666666666666

# ------------------------------------

a = pd.Series([9, 35, 74, 150, 237, 223, 152, 81, 32, 7])

rel_freq = a / a.sum()

print a.sum()

# -------------------------------------

sheet = pd.read_csv('D:\\Source\\Data science & AI\\sheet.tsv', sep='\t', header=None)
avg_karmas_per_posts = sheet.iloc[:, 0]
avg_karmas_per_posts.describe()
wrong_mean = avg_karmas_per_posts.sum()/(avg_karmas_per_posts.count()-1)
print wrong_mean
perc = np.percentile(avg_karmas_per_posts.values, '95', interpolation='midpoint') # 20.3423
print perc

print (perc - avg_karmas_per_posts.mean())/avg_karmas_per_posts.std(ddof=1) # 1.5523
# != 1.645

# -------------------------------------

pop = np.array([1,2,3,4])
sample_means = np.array([1, 1.5, 2, 2.5, 1.5, 2, 2.5, 3, 2, 2.5, 3, 3.5, 2.5, 3, 3.5, 4])

# Probability of value be >= 3
print 100-stats.percentileofscore(sample_means, 3.0, kind='strict') # 37.5

print (((pop - pop.mean()) ** 2).sum() / len(pop)) ** 0.5 # 1.11803398875
print pop.std() # 1.11803398875

print sample_means.std() # 0.790569415042

print pop.std() / sample_means.std() # 1.4142
print 2 ** .5 # 1.4142

# -----------------------------------

pop = np.array([1,2,3,4,5,6])
print pop.mean(), ' ', pop.std()

print pop.std() / 2**.5

print 3.49 / 5 ** .5

# -----------------------------------

klout_scores = pd.read_csv('D:\\Source\\experiments\\Data science & AI\\Klout scores.csv', header=None)
klout_scores.describe()
klout_scores = klout_scores.iloc[:,0]
klout_scores.mean() # 37.719054832538156
klout_scores.std(ddof=0) # 16.036658421715316

# Plot klout_scores
plt.hist(klout_scores, bins=20)
plt.show()

# Plot means of 100 random samples of size 35
import random
samples = pd.DataFrame()
for i in range(100):
    sample = random.sample(klout_scores.values, 35)
    samples['Sample ' + str(i+1)] = sample
pass
samples.head()
samples.describe()

sample_means = samples.apply(lambda a: a.mean())
sample_means.describe() 

plt.hist(sample_means, bins='auto')
plt.show()




