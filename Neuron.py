import numpy


class Neuron:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights

    def sum_inputs(self):
        res = 0.0
        for i in range(len(self.inputs)):
            res += inputs[i] * weights[i]
        return res

    def sigmoid(self):
        inp = sum_inputs()
        return 1 / (1 + numpy.exp(-inp))

    def derivative(self, error):
        return error * (1 - error)

    def updateWeights(weights):
        self.weights = weights
