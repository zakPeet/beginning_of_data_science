import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

contributions = np.genfromtxt('F:\Zocuments\contributions.csv', delimiter=',', skip_header=1, usecols=[1, 2])
contributionsMean = np.mean(contributions[:,0])
contributionsMedian = np.median(contributions)
contributionsStd = np.std(contributions)
contributionsMale = np.mean(contributions[:,1] > 1)
print(contributionsMean)

normaldist = np.random.normal(contributionsMean,contributionsStd, size=10000 )

plt.hist(normaldist, range=(275, 375))
plt.xlabel("values")
plt.ylabel("frequencies")
plt.show()