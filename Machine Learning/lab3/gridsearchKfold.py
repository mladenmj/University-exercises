import numpy as np
import pvml

def load_set(filename):
	data = np.loadtxt(filename)
	X = data[:,:-1]
	Y=data[:,-1]
	return X, Y

def kfold_cv(X, Y, k, gamma, lambda_,lr=0.01,steps=200000):
	m = X.shape[0]
	folds = np.arange(m) % k
	np.random.shuffle(folds)
	tot_accuracy = 0
	
	for fold in range(k):
		Xtrain = X[folds != fold, :]
		Ytrain = Y[folds != fold]
		Xval = X[folds == fold, :]
		Yval = Y[folds == fold]
		alpha, b = pvml.ksvm_train(Xtrain,Ytrain,"rbf",gamma, lambda_,lr=0.01,steps=1000)
		labels, logits = pvml.ksvm_inference(Xval, Xtrain, alpha, b, "rbf", gamma)
		accuracy = (labels==Yval).mean() * 100
		tot_accuracy += accuracy
	return tot_accuracy / k
		

###############################################

X, Y = load_set("page-blocks-train.txt")
Xtest, Ytest = load_set("page-blocks-test.txt")

lambdas = [1e-7, 3e-7, 1e-6, 3e-6]
gammas = [1e-7, 3e-7, 1e-6, 3e-6]

best_acc = 0
best_lambda = None
best_gamma = None

for lambda_ in lambdas:
	for gamma in gammas:
		accuracy = kfold_cv(X, Y, 5, gamma, lambda_,lr=0.01,steps=200000)
		print("lambda =", lambda_, "gamma =", gamma, "accuracy =", accuracy)
		if accuracy > best_acc:
			best_acc = accuracy
			best_lambda = lambda_
			best_gamma = gamma

print("TEST")
alpha, b = pvml.ksvm_train(X, Y, "rbf", best_gamma, best_lambda, lr=0.01,steps=200000)
labels, logits = pvml.ksvm_inference(Xtest, X, alpha, b, "rbf", best_gamma)
accuracy = (labels==Ytest).mean() * 100
print("lambda =", lambda_, "gamma =", gamma, "accuracy =", accuracy)



