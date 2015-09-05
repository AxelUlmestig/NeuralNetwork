import math

class Sigmoid:
	def __init__(self, steepness = 1, offset = 0):
		self.steepness = steepness
		self.offset = offset

	def evaluate(self, x):
		expArgument = -self.steepness * (x + self.offset)
		numerator = 1
		denominator = 1 + math.exp(expArgument)
		return numerator/denominator
