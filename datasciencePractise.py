import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#importing file

stats = np.genfromtxt('F:\jobs\SoccerStats.csv', delimiter=',', skip_header=1, usecols=[1, 2])
# the players' means and std
player_means = np.mean(stats, axis=0)
player_std = np.std(stats, axis=0)
# the five number summary
player1_median = np.median(stats[:, 0])
player2_median = np.median(stats[:, 1])
player1_3rd_quartile = np.percentile(stats[:, 0], 75)
player1_3rd_quartile = np.percentile(stats[:, 1], 75)

# testing for which distribution the data for the various players fall
normal1 = np.random.normal(player_means[0], player_std[0], size=10000)
normal2 = np.random.normal(player_means[1], player_std[1], size=10000)
binomial1 = np.random.binomial(54, 0.8186, size=10000)
binomial2 = np.random.binomial(54, 0.78, size=10000)


# using chi square test where where sum of observed minus expected is calculated and tested over chi square with n-1 df, 
   # pick smaples from each of the distributions to compare
normalsample1 = np.random.choice(normal1, size=15, replace=False)
normalsample2 = np.random.choice(normal2, size=15, replace=False)
binomialsample1 = np.random.choice(binomial1, size=15, replace=False)
binomialsample2 = np.random.choice(binomial2, size=15, replace=False)
    # calculate the chi value and compare it to chi (15-1) df = 
i = 0
player1normalvalues=0
player2normalvalues=0
player1binomialvalues=0
player2binomialvalues=0

for i in range(14):
   player1normalvalues += (stats[i,0] - normal1[i])**2/normal1[i]
   player2normalvalues += (stats[i,1] - normalsample2[i])**2/normalsample2[i]
   player1binomialvalues += (stats[i,0] - binomialsample1[i])**2/binomialsample1[i]
   player2binomialvalues += (stats[i,1] - binomialsample2[i])**2/binomialsample2[i]


plt.hist(normal1)
plt.hist(binomial1, bins=30, histtype='step', label='Player1')
plt.hist(binomial2, bins=30, histtype='step', label='player2')
#plt.legend('upper right')
#plt.show()
prob1 = np.mean(stats[:,0] >= 30)
prob2 = np.mean(binomial1 >= 30)
prob = np.mean(normal1 >= 30)
print(player_means)