import numpy as np

def load_data(filename):
	data = np.loadtxt(filename)
	X = data[:, :2].astype(int)
	X[:, 0] -= 1
	Y = data[:, -1]
	
	return X, Y

def nb_train (X, Y):
	k = 2
	q = 3	# one feature has less, but we don't distinguish, simpler
	m, n = X.shape
	priors = np.empty(k)
	probs = np.empty((k, n, q))
	
	for c in range(k):
		mc = (Y == c).sum()
		priors[c] = mc / m
		
		for j in range(n):
			# Select only rows of X where we have class c
			hist = 1 + np.bincount(X[Y == c, j], minlength=q)
			probs[c, j, :] = hist / (mc+q)
			
		return probs, priors

def nb_inference (X, probs, priors):
	m = X.shape[0]
	k, n, q = probs.shape
	scores = np.empty((m, k))
	
	for c in range(k):
		scores[:, c] = -np.log(priors[c])
		
		for j in range(n):
			scores[:, c] -= np.log(probs[c, j, X[:, j]])
	
	return np.argmin(scores, 1)
	
#########################################################

X, Y = load_data("../titanic/titanic-train.txt")

probs, priors = nb_train(X, Y)
#print("Priors:", priors)
#print("Probabilities", probs)

predictions = nb_inference(X, probs, priors)
accuracy = (Y == predictions).mean()

print("Training accuracy:", accuracy * 100)

# Test
X, Y = load_data("../titanic/titanic-test.txt")
predictions = nb_inference(X, probs, priors)
accuracy = (Y == predictions).mean()

print("Test accuracy:", accuracy * 100)

