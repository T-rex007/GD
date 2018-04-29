#! usr/ bin/ python 
"""
Author: Tyrel S.I Cadogan
Email: shaqc777@yahoo.com
Gradient descent Algorithm Implemeted on a single feature from kaggle's housing prices dataset.
"""
import numpy as np 
import pandas as pd
import tensorflow as tf 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_iterations", default = 100, type = int, help = "-The number of iterations for the gradient descent")
parser.add_argument("--learning_rate", default = 0.0001, type = float, help = "-The learning rate")

def loadData(features):
	"""
	Returns a scaled array of the of features specifified by the input parameter feature
	Args:
		features: a list of the feature names
	
	"""
	from sklearn.preprocessing import StandardScaler
	data = pd.read_csv("train.csv")
	dataset = data[features].as_matrix()
	scaler = StandardScaler()
	dataset = scaler.fit_transform(dataset)

	return dataset

def doPlot(b, m, points):
	"""
	Plots a line graph illustrating the model superpositioned over the Datapoints
		Args:
		m : Gradient of the linear model
		b : Y-intercept of linear model
		points : pandas dataframe containg the ouput and input vales to be plotes.
		"""
	import matplotlib.pyplot as plt 
	pred = producePred(b = b, m = m, data = points[0])
	### plot the trained model
	plt.plot(points[0], pred, color = 'r')
	plt.ylabel('Sale Price')
	plt.xlabel('Lot Area')
	plt.scatter(points[0], points[1])
	plt.show()

	return


def computeError(m, b, points):
	"""
	Returns the mean squared error for a griven set of points
		Args:
		m : Gradient of the linear model
		b : Y-intercept of linear model
		points : data to compute error for.


	"""
	error = 0.0
	N = len(points)
	### Compute error for each point
	for i in range(N):
		x = points[i, 0]
		y = points[i, 1]
		#print(x,y, m)
		### Compute error for point
		error += (y  -((m*x) + b))**2 
		
	### Return the mean squared error of the given set of points
	return error / float(N)

def producePred(m, b, data):
	"""
	Returns predictions of the trained model, given model parameters and data

	"""

	pred = []
	for value in data:
		pred.append(m*value + b)
	return pred

def gradientStep(b_current, m_current, points, learning_rate):
	"""
	Returns the updated values of the parameters that describe the linear model in this case 
	m(Gradient of the line) and b( the y-intercept of the line).
	NB: the y-intercept is the point to which the line cuts the y-axis or the output when the input value is zero

	Args:

	    b_current: The current value of b(y-intercept) to be updated
	    m_current: the current values of m(gradient) to be updated
	    points: The data
	    learning_rate: This is a paramenter that describes how huge of a step the algorithm must take
	"""

	### Initializing m and b
	m_gradient = 0.0
	b_gradient = 0.0

	N = float(len(points))
	### Cranking the magic machine........poof
	### Optimizing the parameters of the linear model
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		m_gradient += - (2/N) * x * (y - ((m_current * x )+ b_current))
		b_gradient += - (2/N) * ((y - ((m_current * x) + b_current)))

	### Updating curent values of the parameters
	b_new = b_current - (learning_rate * b_gradient)
	m_new = m_current - (learning_rate * m_gradient)

	return b_new, m_new

def gradientDescentRunner( init_m, init_b, points, n_iterations, learning_rate):
	"""
	Returns the b(y-intercept) and m(gradient) after i given number of iterations stored in the parameter 
	n_iterations
		Args:
			init_m: Intitail gradient value
			init_b: Initial y-intecept value
			points: Inputdata
			n_iterations: how many steps should the algorithm take(the number of iterations)
			learning_rate: How big of a step the algorithm should take( The Learning Rate)

	"""
	m = init_m
	b = init_b

	for x in range(n_iterations):
		b, m = gradientStep(b, m, points, learning_rate)
	return [b, m]


def main(argv):

	args = parser.parse_args(argv[1:])
	feat_list = ['LotArea', 'SalePrice']
	points = loadData(feat_list)
	b_temp = 0.0
	m_temp = 0.0
	
	b, m = gradientDescentRunner(init_b = b_temp, init_m = m_temp, points = points, n_iterations = args.n_iterations, learning_rate = args.learning_rate)
	print()
	print("-The output when the input is zero(y-intercept is", b)
	print("-The rate at which the output changes with respect to the input(gradient) is ", m)
	print("The error is", computeError(m,b, points))
	print()
	doPlot( m = m, b = b, points = pd.DataFrame(points))
	return


if __name__ == "__main__":

	tf.app.run(main) 