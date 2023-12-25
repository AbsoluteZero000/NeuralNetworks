import numpy
import pandas
from sklearn.model_selection import train_test_split
import openpyxl
import random
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



def read_excel_data(file_path, sheet_name="concrete_data"):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook[sheet_name]
    headers = [cell.value for cell in sheet[1]]
    data = []
    for row in sheet.iter_rows(min_row=2):
        row_data = [cell.value for cell in row]
        data.append(row_data)

    return headers, data

def train_test_split(data, target_column_index, test_size=0.25, random_state=None):
    random.seed(random_state)

    # Shuffling the data
    random.shuffle(data)

    # Splitting the data into training and testing sets
    split_index = int(len(data) * (1 - test_size))
    training_data = data[:split_index]
    testing_data = data[split_index:]

    # Extracting features and targets
    features_train = [row[:target_column_index] for row in training_data]
    targets_train = [row[target_column_index] for row in training_data]
    features_test = [row[:target_column_index] for row in testing_data]
    targets_test = [row[target_column_index] for row in testing_data]

    return features_train, features_test, targets_train, targets_test

# Example usage

if __name__ == "__main__":
    file_path = "concrete_data.xlsx"
    headers, data = read_excel_data(file_path)
    target_column_index = 4  # Assuming the target is in the fifth column

    training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(
        data, target_column_index, test_size=0.25, random_state=42
    )

    input_weights, oWeights = initialize_weights(4,3,1)

    learning_rate = 1
    epocs = 100
    output_weights = [item for sublist in oWeights for item in sublist]
    # print(weight)
    stop = False
    while epocs > 0:
        for i in range(len(training_inputs)):
            hidden_layer = constructHiddenLayer(training_inputs[i], input_weights)
            output = calculateOutput(hidden_layer, output_weights)
            predicted_value = output.sigmoid(False, True)
            mean_square_error = (1/2) * pow((training_targets[i] - predicted_value), 2)
            error = (output.sigmoid(True, True) * (training_targets[i] - predicted_value))
            weights = []
            j = 0
            for weight in output.weights:
                weights.append(weight + learning_rate * (error * output.inputs[j]))
                j += 1
            output.updateWeights(weights)
            input_weights = []
            hidden_layer_neurons = hidden_layer.getNeurons()
            # print(error)
            for neuron in hidden_layer_neurons:
                inputweights = []
                j = 0
                for weight in neuron.weights:
                    error_input = (neuron.sigmoid(True, False) * error * (weight))
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
        predicted_value = output.sigmoid(False, True)
        mean_square_error = (1/2) * pow((testing_targets[i] - predicted_value), 2)
        error = (output.sigmoid(True, True) * (testing_targets[i] - predicted_value))
        print("Testcase: ", testing_inputs[i])
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
        predicted_value = output.sigmoid(False, True)
        print("Predicted Value = ", predicted_value)
        print("Acual Value = ", testing_targets[i])
