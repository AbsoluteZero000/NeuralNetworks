import numpy
import pandas
from sklearn.model_selection import train_test_split

from Neuron import Neuron
from Layer import Layer


def initialize_weights(input_size, hidden_size, output_size):
    Wih = numpy.random.randn(input_size - 1 , hidden_size + 1)
    who = numpy.random.randn(hidden_size, output_size)
    return Wih, who


def constructHiddenLayer(trainingInput, weightInput):
    HiddenLayer = Layer()
    inputs = []
    for i in range(3):
        neuron = Neuron(trainingInput, weightInput[i])
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



def extract():
    excel_file = "concrete_data.xlsx"
    df = pandas.read_excel(excel_file, header=0)
    features = df.iloc[:, :4].values
    targets = df.iloc[:, 4].values
    return  train_test_split(
    features, targets, test_size=0.25, random_state=42)

if __name__ == "__main__":
    training_inputs, testing_inputs, training_targets, testing_targets = extract()

    input_weights, oWeights = initialize_weights(4,3,1)
    # print(training_inputs)
    # print(oWeights)

    learning_rate = 1
    epocs = 100
    output_weights = [item for sublist in oWeights for item in sublist]
    # print(weight)
    stop = False
    while epocs > 0:
        for i in range(len(training_inputs)):
            hidden_layer = constructHiddenLayer(training_inputs[i], input_weights)
            output = calculateOutput(hidden_layer, output_weights)
            predicted_value = output.sigmoid()
            mean_square_error = (1/2) * pow((training_targets[i] - predicted_value), 2)
            error = (predicted_value * (1 - predicted_value) * (training_targets[i] - predicted_value))
            weights = []
            j = 0
            for weight in output.weights:
                weights.append(weight + learning_rate * (error * output.inputs[j]))
                j += 1
            output.updateWeights(weights)
            input_weights = []
            hidden_layer_neurons = hidden_layer.getNeurons()
            for neuron in hidden_layer_neurons:
                inputweights = []
                j = 0
                for weight in neuron.weights:
                    error_input = (neuron.sigmoid() * (1 - neuron.sigmoid()) * error * (weight))
                    inputweights.append(weight + learning_rate * (error_input * neuron.inputs[j]))
                    j += 1
                neuron.updateWeights(inputweights)
                input_weights.append(inputweights)
            output_weights = output.weights
        if stop:
            break
        epocs -=1

    print("Testing phase starts")
    for i in range(len(testing_inputs)):
        hidden_layer = constructHiddenLayer(testing_inputs[i], input_weights)
        output = calculateOutput(hidden_layer, output_weights)
        predicted_value = output.sigmoid()
        mean_square_error = (1/2) * pow((testing_targets[i] - predicted_value), 2)
        error = (predicted_value * (1 - predicted_value) * (testing_targets[i] - predicted_value))
        print("Predicted Value = ", predicted_value)
        print("Acual Value = ", testing_targets[i])
        print("Error = ", error)
        print("Mean Square Error = ", mean_square_error)

    choice = input("Enter 1 if you want to enter data, anything else if you want to exit: ")
    if choice == '1':
        inputs = []
        for i in range (4):
            inp = float(input("Enter the input value"))
            inputs.append(inp)
        hidden_layer = constructHiddenLayer(inputs, input_weights)
        output = calculateOutput(hidden_layer, output_weights)
        predicted_value = output.sigmoid()
        print("Predicted Value = ", predicted_value)
        print("Acual Value = ", testing_targets[i])
