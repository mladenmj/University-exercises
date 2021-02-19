import numpy as np
import pvml

def load_set(filename):
	data = np.loadtxt(filename)
	X = data[:,:-1]
	Y=data[:,-1]
	return X, Y


###############################################

X, Y = load_set("page-blocks-train.txt")
X, Y = load_set("page-blocks-val.txt")
X, Y = load_set("page-blocks-test.txt")

lambdas = [1e-7, 3e-7, 1e-6, 3e-6]
gammas = [1e-7, 3e-7, 1e-6, 3e-6]

best_acc = 0
best_lambda = None
best_gamma = None
best_alpha = None
best_b = None

for lambda_ in lambdas:
	for gamma in gammas:
		alpha, b = pvml.ksvm_train(X,Y,"rbf",gamma, lambda_,lr=0.01,steps=1000)
		labels, logits = pvml.ksvm_inference(Xval, X, alpha, b, "rbf", gamma)
		accuracy = (labels==Yval).mean() * 100
		print("lambda =", lambda_, "gamma =", gamma, "accuracy =", accuracy)
		if accuracy > best_acc:
			best_acc = accuracy
			best_lambda = lambda_
			best_gamma = gamma
			best_alpha = alpha
			best_b = b

print("TEST")

labels, logits = pvml.ksvm_inference(Xtest, X, best_alpha, best_b, "rbf", best_gamma)
accuracy = (labels==Ytest).mean() * 100



