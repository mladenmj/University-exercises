import numpy as np
import matplotlib.pyplot as plt
import random

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
        #reg_term = lambd * np.norm(w)		# add in case of L2 reg.
        #return loss+reg_term

def logreg_train(X, Y, Xt, Yt, lr=0.003, steps=5000, init_w=None, init_b=0):
        m, n = X.shape
        w = (init_w if init_w is not None else np.zeros(n))
        b = init_b

        for step in range(steps):
                P = logreg_inference(X, w, b)
                Ltr = cross_entropy(P, Y)
                grad_b = (P - Y).mean()
                grad_w = (X.T @ (P - Y)) / m
                b -= lr * grad_b
                w -= lr * grad_w
                
                if(step % 100==0):
                	#plt.scatter(step, Ltr)
                	Pt = logreg_inference(Xt,w,b)
                	Ltest = cross_entropy(Pt, Yt)
                	#plt.scatter(step, Ltest)
                	print(step, Ltr, Ltest)
	
        return w, b
        
def mbgd(trData, Xt, Yt, lr=0.003, steps=50000, init_w=None, init_b=0):
        m, n = X.shape
        w = (init_w if init_w is not None else np.zeros(n))
        b = init_b

        for step in range(steps):
        	idx = np.random.choice(trData.shape[0], size=500, replace=False)
        	batch = trData[idx, :]
        	Xb = batch[:,:6]
        	Yb = batch[:,6]
        	P = logreg_inference(Xb, w, b)
        	Ltr = cross_entropy(P, Yb)
        	grad_b = (P - Yb).mean()
        	grad_w = (Xb.T @ (P - Yb)) / m
        	b -= lr * grad_b
        	w -= lr * grad_w
        	
        	# Test data analysis:
        	if(step % 100==0):
                	plt.scatter(step, Ltr)
                	Pt = logreg_inference(Xt,w,b)
                	Ltest = cross_entropy(Pt, Yt)
                	#plt.scatter(step, Ltest)
                	print(step, Ltr, Ltest)
	
        return w, b

#################################################################

data_tr = np.loadtxt("titanic-train.txt")
data_test = np.loadtxt("titanic-test.txt")

# X are features described as:
# ticketclass=x0, sex=x1, age=x2, spouse=x3, parent/child=x4, fare=x5
# Y=Survived (S=1, NS=0)
X=data_tr[:,:6]
Y=data_tr[:,6]
Xt=data_test[:,:6]
Yt=data_test[:,6]

w, b = logreg_train(X, Y, Xt, Yt)
plt.show()
P = logreg_inference(X, w, b)
predictions = (P > 0.5)
accuracy = (predictions == Y).mean()

print("ACCURACY:", accuracy*100, "%")

print("Weights w:", w)
print("Bias b:", b)
#plt.show()

# Point 2.1
# Guess about my own data:
X_me = np.array([2, 0, 23, 0, 0, 35])

surv = logreg_inference(X_me, w, b)
print("My probability to survive would be:", np.round(surv * 100, 4), "%")

# Point 2.5
plt.scatter(X[:,0], X[:,1])
plt.show()

np.save('weights', w)
np.save('bias', b)

# Batch GD implementation

w, b = mbgd(data_tr, Xt, Yt)
plt.show()


