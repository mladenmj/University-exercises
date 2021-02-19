import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def logreg_inference (X, w, b):

	logits = (X @ w) + b
	P = sigmoid(logits)
	
	return P

def cross_entropy(P, Y):
	eps = 1e-3
	P = np.clip(P, eps, 1-eps)
	return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()

def logreg_train(X, Y, lr=1e-3, steps=10000, init_w=None, init_b=0):
	m, n = X.shape
	w = (init_w if init_w is not None else np.zeros(n))
	b = init_b
	
	for step in range(steps):
		P = logreg_inference(X, w, b)
		grad_b = (P - Y).mean()
		grad_w = (X.T @ (P - Y)) / m
		b -= lr * grad_b
		w -= lr * grad_w
		
	return w, b

#################################################################

data = np.loadtxt("exam.txt")
X = data[:,:2]
Y = data[:,2]

#plt.scatter(X[:,0], X[:,1], c=Y)
#plt.show()

w, b = logreg_train(X, Y)

P = logreg_inference(X, w, b)
predictions = (P > 0.5)
accuracy = (predictions == Y).mean()

print("ACCURACY:", accuracy*100)

