#A neuron simply maps input -> output while doing some form of math in between
#Summation of each (input * weight) to an output, this happens in a normal linear regression as well 

#What makes neural networks different from simple linear regression is the inclusion of an activation function
#that allows for non linearity

import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        result = np.dot(self.weights, inputs) + self.bias 
        return sigmoid(result)

weights = np.array([0,1])
bias = 4

N = Neuron(weights=weights, bias= bias)

x = np.array([2,3])

print(N.feedforward(x))