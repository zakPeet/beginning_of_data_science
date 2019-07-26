# neural network to find square roots
# for x=1:y=1, x=4:y=2, x=9:y=3
# values of y for expected output are in form of (1/1+exp(-y))

import numpy as np
import matplotlib.pyplot as plt


# inputs

x1 = np.array([[1, 1],
              [1, 4],
              [1, 9],
              [1, 16],
              [1, 25]])

# expected outputs
y = np.array([[0.731058578, 0.880797078, 0.952574126, 0.98201379, 0.993307419]])# 1, 2, 3, 4, 5

# init weights
w1 = np.random.rand(2, 1)
w2 = np.random.rand(1)

# sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))



# learning rate
lr = 0.1

# backpropagation function
def sigmoid_deriv(x):
    return (np.exp(-x))/(1+np.exp(-x))**2




# repetitions, costs and length of x
repetitions = 10000
m = len(x1)
costs = []
bias = np.array([[1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]])


for i in range(repetitions):
    # forward
    node1 = np.matmul(x1, w1)
    act1 = sigmoid(node1) 
    # hidden layer
    node2 = np.matmul(w2, act1.T)
    output = sigmoid(node2)   
    #backprop
    delta1 = y-output
    sigma1 = np.matmul(delta1, sigmoid_deriv(output))
    sigma2 = sigma1*w2*sigmoid_deriv(act1) # where sigma*weights is input for hidden layer
    w1 = w1+lr*(1/m)*np.matmul(bias, sigma2)
    w2 = w2+lr*(1/m)*sigma1
    c = np.mean(np.abs(delta1))
    costs.append(c)
    print(delta1)

    
if i % 100 == 0:
    print("iteration and error:")
    print(i, c)
    
    
    

# plot costs
plt.plot(costs)
plt.show()


print("w1=", w1)
print("w2=", w2)
# testing how much the network has learnt
