import numpy as np
import pvml

def ovr_train(X, Y, k):
	W = np.zeros((2, 3))
	b = np.zeros(3)
	
	for c in range(k):
		Y1 = (Y == c)
		W[:,c], b[c] = pvml.svm_train(X, Y1, 0, lr=0.1, steps=100000)
	return W, b
		
def ovr_inference(X, W, b):
	logits = X @ W + b.T
	return logits.argmax(1)

X,Y = pvml.load_dataset("iris")
k=3
W, b = ovr_train(X, Y, k)
print("Training finished")

predictions = ovr_inference(X, W, b)
accuracy = (predictions == Y).mean() * 100
print("Accuracy:", accuracy)
