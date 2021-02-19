import numpy as np
import matplotlib.pyplot as plt
import pvml

def load_data(filename):
	data = np.loadtxt(filename)
	X = data[:, :-1]
	Y = data[:, -1].astype(int)

	return X, Y

def plot_acc(train_acc, val_acc):
	plt.clf()
	plt.plot(train_accs)
	plt.plot(val_acc)
	plt.legend(["Train", "Validation"])
	plt.xlabel("Epocs")
	plt.ylabel("Accuracy (%)")
	plt.pause(0.01)

#################################################

# Gather data:
X, Y = load_data("train.txt.gz")
Xval, Yval = load_data("validation.txt.gz")
Xtest, Ytest = load_data("test.txt.gz")

# Normalization:
# Mean-var
mean = X.mean(0, keepdims=True)
stddev = X.std(0, keepdims=True)

X = (X - mean) / (stddev + 1e-8)
Xval = (Xval - mean) / np.maximum(stddev, 1e-8)
Xtest = (Xtest - mean) / (stddev + 1e-8)

# Net training and validation:
train_accs = []
val_accs = []

plt.ion()

net = pvml.MLP([1024, 512, 10])

for epoc in range(1000):
	net.train(X, Y, lr=0.0003, steps=1760 // 80, batch=80)	# Training
	labels, probs = net.inference(X)
	accuracy = (labels == Y).mean()
	print("Train accuracy", accuracy * 100)
	train_accs.append(accuracy * 100)
	
	labels, probs = net.inference(Xval)		# Validation
	accuracy = (labels == Yval).mean()
	print("Validation accuracy", accuracy * 100)
	val_accs.append(accuracy * 100)
	
	# Plotting
	plot_acc(train_accs, val_accs)

plt.ioff()
plt.show()

net.save("network.npz")
np.save('labels', labels)	# Saved for error analysis
np.save('probs', probs)
