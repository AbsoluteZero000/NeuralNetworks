import Neuron


class Layer:
    def init(self):
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate) 

    def getNeurons():
        return neurons
