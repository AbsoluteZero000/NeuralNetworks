import numpy
from scipy.special import expit


class Neuron:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights
        self.output = None
        self.delta = None

    def sum_inputs(self):
        res = 0.0
        for i in range(len(self.inputs)):
            res += self.inputs[i] * self.weights[i]
        return res

    def sigmoid(self, deriv=False, output=False):
        inp = self.sum_inputs()
        if output:
            if deriv:
                return 1.0
            return inp
        else:
            if not deriv:
                return expit(inp)
            return expit(inp) * (1 - expit(inp))
    
    def updateWeights(self, weights):
        self.weights = weights
