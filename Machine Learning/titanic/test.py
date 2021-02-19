import numpy as np
import matplotlib.pyplot as plt

def inference(X, w, b):
	z = (X @ w) + b
	p = 1 / (1 + np.exp(-z))
	return p

def cross_entropy(P, Y):
	eps = 1e-3
	P = np.clip(P, eps, 1-eps)
	return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()
        

#######################################################

# Run only after 'titanic.py'!
data_test = np.loadtxt("titanic-test.txt")
X = data_test[:,:6]
Y = data_test[:,6]

# Get weights and bias from training:
w = np.load('weights.npy')
b = np.load('bias.npy')

# Count number of correct guesses to determine accuracy:
P = inference(X, w, b)
predictions = (P > 0.5)
acc_test = (predictions == Y).mean()

print("Test accuracy is:", np.round(acc_test*100, 4), "%")

