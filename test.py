import unittest 
from sigmoid import Sigmoid 
import random
from neuron import Neuron, InputNeuron, WeightedNeuron
class SigmoidTest(unittest.TestCase):
	def testOffset(self):
		offset = random.randint(0,10)
		steepness = random.randint(1,10)
		sigmoid = Sigmoid(steepness, offset)
		
		offsetValue = sigmoid.evaluate(-offset)
		offsetTarget = 1/2
		targetRatio = offsetValue / offsetTarget
		
		self.assertLess(targetRatio, 1.1)
		self.assertGreater(targetRatio, 0.9)

	def testSteepness(self):
		sigmoid = Sigmoid()
		
		inp = random.random()
		greaterInp = inp * 1.1
		
		value = sigmoid.evaluate(inp)
		greaterValue = sigmoid.evaluate(greaterInp)
		self.assertGreater(greaterValue, value)


class InputNeuronTest(unittest.TestCase):
	def testInputNeuronConstructor(self):
		neuron = InputNeuron()
		expectedValue = 0
		self.assertEqual(expectedValue, neuron.evaluate())

	def testInputNeuronValue(self):
		neuron = InputNeuron()
		expectedValue = random.randint(0,10)
		neuron.setValue(expectedValue)
		self.assertEqual(expectedValue, neuron.evaluate())

class NeuronTest(unittest.TestCase):
	def testEmptyEvaluate(self):
		neuron = Neuron([])
		expectedValue = 1/2
		self.assertEqual(expectedValue, neuron.evaluate())

	def testSinglePreviousEvaluate(self):
		previousNeuron = InputNeuron()
		previousNeuron.setValue(1)
		previousRow = [previousNeuron]
		
		neuron = Neuron(previousRow)
		self.assertGreater(neuron.evaluate(), 1/2)

		
class WeightedNeuronTest(unittest.TestCase):
	def testEvaluate(self):
		neuron = InputNeuron()
		neuronValue = random.randint(0,10)
		neuron.setValue(neuronValue)
		
		coefficient = random.randint(0,10)
		weightedNeuron = WeightedNeuron(neuron, coefficient)

		expectedValue = neuronValue * coefficient
		self.assertEqual(expectedValue, weightedNeuron.evaluate())
	
		

if __name__ == '__main__':
	unittest.main()
	
