import pvml
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
	data = np.loadtxt(filename)

	X = data[:, :-1]
	Y = data[:, -1].astype(int)

	return X, Y

def confmat_plot(cm):
	x = np.arange(cm.shape[0])
	y = np.arange(cm.shape[1])

	# Plotting:
	fig, ax = plt.subplots()
	im = ax.imshow(cm, cmap="jet")

	# Show all ticks and labels:
	ax.set_xticks(np.arange(len(x)))
	ax.set_yticks(np.arange(len(y)))
	ax.set_xticklabels(x)
	ax.set_yticklabels(y)

	# Colorbar:
	cbar = ax.figure.colorbar(im, ax=ax)

	# Set ticks position
	ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)

	# Turn spines off and create white grid
	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
	ax.tick_params(which="minor", bottom=False, left=False)

	# Print data inside the heatmap-matrix
	for i in range(len(x)):
	    for j in range(len(y)):
	        text = ax.text(j, i, cm[i, j].round(3),
	                       ha="center", va="center", color="w")

	ax.set_title("Confusion matrix\n", fontweight='bold')
	fig.tight_layout()

def confmat(labels, Yval):
	cm = np.zeros((10,10))

	for lbl, y in zip(labels, Yval):
		cm[lbl, y] += 1

	cm /= cm.sum(1, keepdims=True)

	# Rounding numbers to 3 decimal positions:
	tmp_rnd = cm.round(3)

	# Alter most important result by a very small quantity (~1e-5) to get
	# sum of rows equal to 1. Useful for plotting data in heatmap

	for i in range(cm.shape[0]):
		if(tmp_rnd.sum(1)[i] != 1):
			tmp_rnd[i, i] += 1-tmp_rnd.sum(1)[i]

	cm = tmp_rnd * 100

	confmat_plot(cm)

	return cm

def mispredict(Xval, Yval, probs, labels):
	m = Xval.shape[0]
	# pcorrect[0] = probs[0, Yval[0]]
	# pcorrect[1] = probs[1, Yval[1]]
	# ...
	pcorrect = probs[np.arange(m), Yval]
	idx = pcorrect.argsort()

	names = open("validation-names.txt").read().split()

	print("* Mispredictions *")
	print("\nFilename - Probability estimate - Actual number - Predicted number:")

	for i in range(10):
		print(names[idx[i]], pcorrect[idx[i]], Yval[idx[i]], labels[idx[i]])

####################################################

# Data and net loading:
X, Y = load_data("train.txt.gz")
Xval, Yval = load_data("validation.txt.gz")
Xtest, Ytest = load_data("test.txt.gz")

net = pvml.MLP.load("network.npz")
labels = np.load("labels.npy")
probs = np.load("probs.npy")

# Error analysis

# Confusion matrix:
cm = confmat(labels, Yval)

#print("\nConfusion matrix:")
#print(cm)
np.save('confusion_matrix', cm)

# Misclassification analysis:
mispredict(Xval, Yval, probs, labels)

# Plotting spectrograms for each digit:
w = net.weights[0]
max_weight = np.abs(w).max()
#print(w.shape)
plt.figure()

for i in range(10):
	plt.subplot(5, 2, i+1)

	# reshape as spectrograms
	plt.imshow(w[:, i].reshape(16, 64), cmap="seismic", vmin=-max_weight, vmax=max_weight)

plt.show()
