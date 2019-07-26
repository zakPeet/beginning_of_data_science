# neural network to solve XOR problem
# 1 1 -> 0
# 1 0 - >1
# 0 1 -> 1
# 0 0 -> 0

import numpy as np
import matplotlib.pyplot as plt

# inputs
x = np.array([[1, 1, 1],
              [1, 1, 0],
              [0, 1, 1],
              [0, 0, 0]])

# expected outputs
y = np.array([[0, 1, 1, 0]])




# learning rate, costs, length of input, repetitions, cost
lr = 0.1
costs = []
m = len(x)
repetitions = 10000
costs = []
bias = np.array([[1],
                 [1],
                 [1]])
bias2 = np.array([[1],
                 [1],
                 [1],
                 [1]])

#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# backpropagation function
def sigmoid_deriv(x):
    return (np.exp(-x))/((1+np.exp(-x))**2)

#init  weights
w1 = np.random.rand(3,1)
w2 = np.random.rand(3,1)
w3 = np.random.rand(3,1)
w4 = 1
w5 = 1
w6 = 1
w7 = 1

# training process
for i in range(repetitions):
    # forward
    node1 = np.matmul(x, w1)
    node2 = np.matmul(x, w2)
    node3 = np.matmul(x, w3)
    act1i = sigmoid(node1)
    act2i = sigmoid(node2)
    act3i = sigmoid(node3)
    #   hidden layer
    h1 = w4*act1i
    h2 = w5*act2i
    h3 = w6*act3i
    actj = sigmoid(h1+h2+h3)
    O1 = w7*actj
    actk = sigmoid(O1)
    
    # backpropagtion
    delta1 = y-actk.T
    # output layer
    sigma1 = np.matmul(sigmoid_deriv(actk).T, delta1.T)
    # hidden layer
    sigma21 = w7*sigma1*sigmoid_deriv(actj)
    sigma22 = w7*sigma1*sigmoid_deriv(actj)
    sigma23 = w7*sigma1*sigmoid_deriv(actj)
    # input layer
    interlude1 = np.matmul(sigma21, bias.T)
    sigma211 = w4*np.matmul(interlude1.T, sigmoid_deriv(act1i))
    interlude2 = np.matmul(sigma22, bias.T)
    sigma221 = w5*np.matmul(interlude2.T, sigmoid_deriv(act2i)) 
    interlude3 = np.matmul(sigma23, bias.T)
    sigma231 = w6*np.matmul(interlude3.T, sigmoid_deriv(act3i)) 
    # adjust weights
    w1 = w1 + lr*(1/m)*sigma211                    # add because they are matrices,minimizes error 
    w2 = w2 - lr*(1/m)*sigma221                    # add because they are matrices,minimizes error
    w3 = w3 - lr*(1/m)*sigma231                    # add because they are matrices, minimizes error
    w4 = w4 + lr*(1/m)*np.matmul(sigma21.T, bias2) #subtract because they are numbers,minimizes error
    w5 = w5 + lr*(1/m)*np.matmul(sigma22.T, bias2) #subtract because they are numbers,minimizes error
    w6 = w6 - lr*(1/m)*np.matmul(sigma23.T, bias2) #subtract because they are numbers,minimizes error
    w7 = w7 - lr*(1/m)*sigma1
    
    c = np.mean(np.abs(delta1))
    costs.append(c)
#plotting
plt.plot(costs)
plt.show()
    
    
    


