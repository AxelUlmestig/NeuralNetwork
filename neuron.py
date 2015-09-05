from sigmoid import Sigmoid
import random

class Neuron:
	
	def __init__(self, previousRow):
		self.weightedNeurons = getWeightedNeurons(previousRow)
		self.sigmoid = Sigmoid()

	def evaluate(self):
		sum = 0
		for weightedNeuron in self.weightedNeurons:
			sum += weightedNeuron.evaluate()
		if len(self.weightedNeurons) > 0:
			sum /= len(self.weightedNeurons)
		return self.sigmoid.evaluate(sum)


class InputNeuron:
	
	def __init__(self):
		self.value = 0

	def evaluate(self):
		return self.value
	
	def setValue(self, value):
		self.value = value

class WeightedNeuron:

	def __init__(self, neuron, coefficient):
		self.neuron = neuron
		self.coefficient = coefficient

	def evaluate(self):
		return self.coefficient * self.neuron.evaluate()

	def getCoefficient(self):
		return self.coefficient

def getWeightedNeurons(row):
	weightedNeurons = []
	for neuron in row:
		coefficient = random.random()
		weightedNeuron = WeightedNeuron(neuron, coefficient)
		weightedNeurons.append(weightedNeuron)
	return weightedNeurons
