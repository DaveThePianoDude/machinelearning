#!/usr/bin/env python


import numpy as np

import matplotlib.pyplot as plt

def perceptron(X,Y):

    n, d = X.shape
    new_col = np.ones((n,1))
    #X = np.concatenate((X,new_col),1)

    W = np.zeros(d)
    b = 0

    max = 1000
    iter = 0
    done = False

    #X = np.random.permutation(X)
    while iter < max and not done:
        print(iter)
        iter = iter + 1
        #X = np.random.permutation(X)
        clear = True
        for index in range(0,len(X)):
            print(X[index])
            activator = np.dot(W.T,X[index]) + b
            if activator * Y[index] <= 0:
                W = W + Y[index] * X[index]
                b = b + Y[index]
                clear = False
                #print (Y[index])
        print(clear)
        if clear:
            done = True
            print(W)
            print(b)

    return (W,b)

N = 100;
d = 10;
x = np.random.rand(N,d)
w = np.random.rand(1,d)
y = np.sign(w.dot(x.T))[0]
w, b = perceptron(x,y)

print ("GOT HERE AFTER TEST 1")
print (b)

x = np.array([ [-0.70072, -1.15826],  [-2.23769, -1.42917],  [-1.28357, -3.52909],  [-3.27927, -1.47949],  [-1.98508, -0.65195],  [-1.40251, -1.27096],  [-3.35145,-0.50274],  [-1.37491,-3.74950],  [-3.44509,-2.82399],  [-0.99489,-1.90591],   [0.63155,1.83584],   [2.41051,1.13768],  [-0.19401,0.62158],   [2.08617,4.41117],   [2.20720,1.24066],   [0.32384,3.39487],   [1.44111,1.48273],   [0.59591,0.87830],   [2.96363,3.00412],   [1.70080,1.80916]])
y = np.array([1]*10 + [-1]*10)
w, b = perceptron(x,y)

print ("GOT HERE AFTER TEST 2")

def classify(w,xs,b):
	#xs[0] = 1
	#print ("Here is W:")
	if np.dot(w,xs) + b <= 0:
		return -1
	else:
		return 1

def classify_linear(xs,w,b=None):

    w = w.flatten()
    print(xs)
    predictions=np.zeros(xs.shape[0])

    index = 0
    for item in xs:
        predictions[index] = classify(w,item,b)
        index = index + 1

    return predictions

def test_linear1():
    xs = np.random.rand(50000,20)-0.5 # draw random data
    w0 = np.random.rand(20)
    b0 =- 0.1 # with bias -0.1
    ys = classify_linear(xs,w0,b0)
    print ("Here are the ys:")
    print(ys)
    #uniquepredictions=np.unique(ys) # check if predictions are only -1 or 1
    #print(uniquepredictions)
    return True
    #return set(uniquepredictions)==set([-1,1])

print ("Result of TEST LINEAR 1:")
test_linear1()

# x axis values
#x = [1,-2,3]
# corresponding y axis values
#y = [2,4,1]
xp, yp = x.T
plt.scatter(xp[:10],yp[:10],c='coral')
plt.scatter(xp[10:],yp[10:],c='lightblue')

# naming the x axis
plt.xlabel('x - axis')

# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('eCornell Perceptron Problem (Positive examples in orange, negative in blue.)')

# function to show the plot
plt.show()
