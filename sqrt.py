import matplotlib.pyplot as plt
import numpy as np

# program to do square roots
x = np.array([[25],
                 [36],
                 [49],
                 [64],
                 [81],
                 [100],
                 [121],
                 [144],
                 [169],
                 [196]])
# expected output
y = np.array([[5],
              [6],
              [7],
              [8],
              [9],
              [10],
              [11],
              [12],
              [13],
              [14]])

# activation function
def act(x):
    return x**(0.7547)

#backpropagation function
def derive_act(x):
    return (0.7547)*x**(-0.2453)

#initialization of variables
w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1
w6 = 1
w7 = 1
w8 = 1
w9 = 1
lr = 0.00009 # learning rate
bias = np.array([[1],
                 [1],
                 [1],
                 [1],
                 [1],
                 [1],
                 [1],
                 [1],
                 [1],
                 [1]])
m = len(x)
repetitions = 20000
costs = []

#training
for i in range(repetitions):
    #forward propagation
    #first layer to hidden
    node1 = act(bias + w1*x + w2*x)
    node2 = act(bias + w3*x + w4*x)
    node3 = act(bias + w5*x + w6*x)
    #hidden to outer layer
    lastnode = act(bias + w7*node1 + w8*node2 + w9*node3)
    delta = y**2-lastnode**2
    #backpropagation
    sigma1 = np.matmul(delta.T, derive_act(lastnode))
    sigma21 = w7*np.matmul(np.matmul(sigma1, derive_act(node1).T), bias)
    sigma22 = w8*np.matmul(np.matmul(sigma1, derive_act(node2).T), bias)
    sigma23 = w9*np.matmul(np.matmul(sigma1, derive_act(node3).T), bias)
    # weight adjustments
    w7 = w7 + lr*(1/m)*sigma1
    w8 = w8 + lr*(1/m)*sigma1
    w9 = w9 + lr*(1/m)*sigma1
    w1 = w1 + lr*(1/m)*sigma21
    w2 = w2 + lr*(1/m)*sigma21
    w3 = w3 + lr*(1/m)*sigma22
    w4 = w4 + lr*(1/m)*sigma22
    w5 = w5 + lr*(1/m)*sigma23
    w6 = w6 + lr*(1/m)*sigma23
    c = np.mean(np.abs(delta))
    costs.append(c)
    
# plotting
print(lastnode)
plt.plot(costs)
plt.show()

# predicting
learning1 = float(input("Enter first value"))
learning2 = float(input("Enter second value"))
learning3 = float(input("Enter third value"))
learning4 = float(input("Enter fourth value"))
learning5 = float(input("Enter fifth value"))
learning6 = float(input("Enter fifth value"))
learning7 = float(input("Enter fifth value"))
learning8 = float(input("Enter fifth value"))
learning9 = float(input("Enter fifth value"))
learning10 = float(input("Enter fifth value"))

z = np.array([[learning1],
              [learning2],
              [learning3],
              [learning4],
              [learning5],
              [learning6],
              [learning7],
              [learning8],
              [learning9],
              [learning10]])
test1 = act(bias + w1*z + w2*z)
test2 = act(bias + w3*z + w4*z)
test3 = act(bias + w5*z + w6*z)
#hidden to outer layer
lasttest = act(bias + w7*test1 + w8*test2 + w9*test3)
print(lasttest)