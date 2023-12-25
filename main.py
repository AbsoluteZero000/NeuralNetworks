import numpy
import pandas
from sklearn.model_selection import train_test_split

from Layer import Layer
from Neuron import Neuron


def initialize_weights(input_size, hidden_size, output_size):
    Wih = numpy.random.randn(input_size, hidden_size)
    who = numpy.random.randn(hidden_size, output_size)
    return Wih, who


def constructHiddenLayer(trainingInput, weightInput):
    HiddenLayer = Layer()
    inputs = []
    for i in range(3):
        neuron = Neuron(trainingInput, weightInput)
        HiddenLayer.add_neuron(neuron)
    return HiddenLayer


def feedForwardHiddenLayer(hiddenLayer):
    neurons = hiddenLayer.getNeurons()
    outputs = []
    for neuron in neurons:
        outputs.append(neuron.sigmoid())
    return outputs


def calculateOutput(hiddenLayer, prevWeights):
    outs = feedForwardHiddenLayer(hiddenLayer)
    output = Neuron(outs, prevWeights)
    return output


def backpropagation(inputs, targets, hidden_layer, output_neuron, learning_rate):
    # bn7sb l outputlayer delta
    output_delta = (output_neuron.output-targets)*output_neuron.output*(1 -output_neuron.output)

    # bn update l output layer weights
    output_neuron.delta = output_delta
    output_neuron.update_weights(learning_rate)

    # n7sb l hidden layer delta
    hidden_deltas = [neuron.weights*output_delta*neuron.output*(1-neuron.output) for neuron in hidden_layer.neurons]

    # bn update l hidden layer weights
    for i, neuron in enumerate(hidden_layer.neurons):
        neuron.delta = hidden_deltas[i]
        neuron.update_weights(learning_rate)

excel_file = "concrete_data.xlsx"
df = pandas.read_excel(excel_file, header=0)
features = df.iloc[:, :4].values
targets = df.iloc[:, 4].values
training_inputs, testing_inputs, training_targets, testing_target = train_test_split(
    features, targets, test_size=0.25, random_state=42
)

weight, oWeights = initialize_weights(4,3,1)
weights = [weight] * 4

outputWeights = [oWeights] * 3

learning_rate = 0.01
hidden_layer = constructHiddenLayer(training_inputs[0], weights)
for i in range(len(training_inputs)):
    hidden_layer = constructHiddenLayer(training_inputs[i], hidden_layer.getNeurons()[i].weights)
    output = calculateOutput(hidden_layer, outputWeights)
    # backpropagation(training_inputs[i], training_targets[i], hidden_layer, output, learning_rate)
    outputWeights = output.weights



epoch = 150
while epoch > 0:
    for i in range(len(training_inputs)):
        hidden_layer = constructHiddenLayer(training_inputs[i], hidden_layer.getNeurons()[i].weights)
        output = calculateOutput(hidden_layer, outputWeights)
        # backpropagation(training_inputs[i], training_targets[i], hidden_layer, output, learning_rate)
        outputWeights = output.weights
    epoch -= 1





for i in range(len(testing_inputs)):
    hidden_layer = constructHiddenLayer(testing_inputs[i], hidden_layer.getNeurons()[i].weights)
    output = calculateOutput(hidden_layer, outputWeights)
    print("Actual Output: ", output.sigmoid())
    print("Target Output: ", testing_target[i])
    print("Error: ", abs(output.sigmoid() - testing_target[i]))
    print("Mean Square Error: ", pow(abs(output.sigmoid() - testing_target[i]), 2))



predictions = []
for i in range(len(testing_inputs)):
    hidden_layer = constructHiddenLayer(testing_inputs[i], hidden_layer.getNeurons()[i].weights)
    output = calculateOutput(hidden_layer, outputWeights)
    predictions.append(output.sigmoid())

# Display predictions
print("Predicted Outputs:", predictions)
