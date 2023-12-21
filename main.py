import Layer
import pandas
import numpy
from sklearn.model_selection import train_test_split

excel_file = "concrete_data.xlsx"
df = pandas.read_excel(excel_file, header=0)
features = df.iloc[:, :4].values
targets = df.iloc[:, 4].values
trainig_inputs, testing_inputs, training_targets, testing_target = train_test_split(
    features, targets, test_size=0.25, random_state=42
)


def initialize_weights(input_size, hidden_size, output_size):
    Wih = numpy.random.randn(input_size, hidden_size)
    who = numpy.random.randn(hidden_size, output_size)
    return Wih, who


def constructHiddenLayer(trainingInput):
    weightInput, weightHidden = initialize_weights(4, 3, 1)
    HiddenLayer = Layer()
    inputs = []
    weights = []
    for value in trainingInput:
        inputs.append(value)
        weights.append(weightInput)
    for i in range (3):
        neuron = Neuron(inputs, weights) 
        HiddenLayer.addNeuron(neuron)
    return HiddenLayer, weightHidden

def feedForwardHiddenLayer(hiddenLayer):
    neurons = hiddenLayer.getNeurons()
    outputs = []
    for neuron in neurons:
        outputs.append(neuron.sigmoid())
    return outputs