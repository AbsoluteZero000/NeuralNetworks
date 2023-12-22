import numpy

class Neuron:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights
        self.output = None
        self.delta = None

    def sum_inputs(self):
        res = 0.0
        for i in range(len(self.inputs)):
            res += inputs[i] * weights[i]
        return res

    def sigmoid(self):
        inp = sum_inputs()
        return 1 / (1 + numpy.exp(-inp))

    def updateWeights(weights):
        self.weights = weights
