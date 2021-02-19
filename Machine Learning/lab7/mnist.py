import numpy as np
import pvml
import matplotlib.pyplot as plt

Xtrain, Ytrain = pvml.load_dataset("mnist_train")
Xtest, Ytest = pvml.load_dataset("mnist_test")

net = pvml.MLP([784, 128, 64, 10])

plt.ion()

train_accs = []
test_accs = []

for epoc in range(100):
	net.train(Xtrain, Ytrain, lr=1e-4, lambda_=1e-5, momentum=0.99, steps=3000, batch=20)
	predictions = net.inference(Xtrain)[0]
	train_acc = (predictions==Ytrain).mean()
	predictions = net.inference(Xtest)[0]
	test_acc = (predictions==Ytest).mean()
	print(train_acc*100, test_acc*100)
	train_accs.append(train_acc*100)
	test_accs.append(test_acc*100)
	
	plt.clf()
	plt.plot(train_accs)
	plt.plot(test_accs)
	plt.legend(["train", "test"])
	plt.xlabel("epocs")
	plt.ylabel("accuracy (%)")
	plt.pause(0.05)
	
	# We can understand where the NN makes errors using this matrix;
	# We see where the net classifies a "zero" with a "one", "two"...
	confusion_matrix = np.zeros((10,10))
	
	for p, y in zip(predictions, Ytest):
		for j in range(10):
			print(confusion_matrix[i, j], end="\t")
		print()
	
plt.ioff()
plt.show()

## Reached minute 37 of video!
