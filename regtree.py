#!/usr/bin/env python

import time

import numpy as np

import matplotlib.pyplot as plt

import numpy as np
from pylab import *
from numpy.matlib import repmat
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import sys

def spiraldata(N=300):
    r = np.linspace(1,2*np.pi,N) # generate a vector of "radius" values
    xTr1 = np.array([np.sin(2.*r)*r, np.cos(2*r)*r]).T # generate a curve that draws circles with increasing radius
    xTr2 = np.array([np.sin(2.*r+np.pi)*r, np.cos(2*r+np.pi)*r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1])*0.2

    # Now sample alternating values to generate the test and train sets.
    xTe = xTr[::2,:]
    yTe = yTr[::2]
    xTr = xTr[1::2,:]
    yTr = yTr[1::2]

    return xTr, yTr, xTe, yTe

xTrSpiral, yTrSpiral, xTeSpiral, yTeSpiral = spiraldata(150)



plt.scatter(xTrSpiral[:,0], xTrSpiral[:,1],30,yTrSpiral)
#plt.show()y_bar

y_bar = 0

getSqrVar = lambda y, yBar: (y-yBar) ** 2

def sqimpurity(yTr):

    N, = yTr.shape
    assert N > 0 # must have at least one sample
    yBar = np.sum(yTr) / N

    # elegant way to apply lambda function to vector components
    return np.sum(np.array([getSqrVar(y,yBar) for y in yTr]))

def sqimpurity_test1():
    yTr = np.random.randn(100) # generate random labels
    impurity = sqimpurity(yTr) # compute impurity
    return np.isscalar(impurity)  # impurity should be scalar

def sqimpurity_test2():
    yTr = np.random.randn(100) # generate random labels
    impurity = sqimpurity(yTr) # compute impurity
    return impurity >= 0 # impurity should be nonnegative

def sqimpurity_test3():
    yTr = np.ones(100) # generate an all one vector as labels
    impurity = sqimpurity(yTr) # compute impurity
    return np.isclose(impurity, 0) # impurity should be zero since the labels are homogeneous

def sqimpurity_test4():
    yTr = np.arange(-5, 6) # generate a vector with mean zero
    impurity = sqimpurity(yTr) # compute impurity
    sum_of_squares = np.sum(yTr ** 2)
    return np.isclose(impurity, sum_of_squares) # with mean zero, then the impurity should be the sum of squares

def sqimpurity_test5():
    yTr = np.random.randn(100) # generate random labels
    impurity = sqimpurity(yTr)
    impurity_grader = sqimpurity_grader(yTr)
    return np.isclose(impurity, impurity_grader)

def squareMe(x,c):
    return x+x+c

yTr = np.random.randn(10)*10
#print(yTr)
c = 100
val = np.apply_along_axis(squareMe,0,yTr,c)
#print(val)

print(sqimpurity_test1())
print(sqimpurity_test2())
print(sqimpurity_test3())
print(sqimpurity_test4())
#sqimpurity_test5()

xor4 = np.array([[1,1,1,1],[1,1,1,0],[1,1,0,1],[1,1,0,0],[1,0,1,1],[1,0,1,0],[1,0,0,1],[1,0,0,0],[0,1,1,1],[0,1,1,0],[0,1,0,1],[0,1,0,0],[0,0,1,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
yor4 = np.array([1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1])

def hasDupe(arr):
    u, c = np.unique(arr, return_counts=True)
    print ("arr=")
    print (arr)
    print ("u=" + str(u))
    print ("c=" + str(c))
    return u[c > 1].any()

def sqsplit(xTr, yTr):

    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples

    bestloss = np.inf
    feature = np.inf
    cut = np.inf

    for featureIndex in range(D):
        print(xTr)
        #print("FEATURE INDEX = " + str(featureIndex))
        # Get the average feature value of this feature.
        avgFeatureValue = 0.5 #np.sum(xTr[:,featureIndex])/N
        # Use it to make the cut.
        Sl_yBar = 0
        Sr_yBar = 0
        leftIndices = set()
        rightIndices = set()

        print("Let's try to partition on the value " + str(avgFeatureValue))

        # Compute the average label for each child
        for xTrRowIndex in range(N):
            print("How does " + str(xTr[xTrRowIndex][featureIndex]) + "compare?" )
            if xTr[xTrRowIndex][featureIndex] <= 0.5:
                print("It's less than or equel")
                leftIndices.add(xTrRowIndex)
                Sl_yBar += yTr[xTrRowIndex]
            else:
                print("It's greater")
                rightIndices.add(xTrRowIndex)
                Sr_yBar += yTr[xTrRowIndex]
        #print(avgFeatureValue)
        #print(len(leftIndices))
        #print(len(rightIndices))
        # Calculate the yBars

        Sl_yBar /= len(leftIndices)
        Sr_yBar /= len(rightIndices)

        leftImpurity = 0
        rightImpurity = 0

        for index in range(N):
            if index in leftIndices:
                leftImpurity += ((yTr[index]-Sl_yBar) ** 2)
            else:
                rightImpurity += ((yTr[index]-Sr_yBar) ** 2)

        loss = leftImpurity + rightImpurity

        #print ("LOSS = " + str(loss))

        if loss <= bestloss:
            bestloss = loss
            cut = avgFeatureValue
            feature = featureIndex

        #print("BEST LOSS = " + str(bestloss))
        #print("CUT = " + str(cut))

    return feature, cut, bestloss
# The tests below check that your sqsplit function returns the correct values for several different input datasets

t0 = time.time()
#fid, cut, loss = sqsplit(xTrIon,yTrIon)
t1 = time.time()

#print('Elapsed time: {:0.2f} seconds'.format(t1-t0))
#print("The best split is on feature 2 on value 0.304")
#print("Your tree split on feature %i on value: %2.3f \n" % (fid,cut))

def sqsplit_test1():
    a = np.isclose(sqsplit(xor4, yor4)[2] / len(yor4), .25)
    #b = np.isclose(sqsplit(xor3, yor3)[2] / len(yor3), .25)
    #c = np.isclose(sqsplit(xor2, yor2)[2] / len(yor2), .25)
    #return a and b and c
    return a

print(sqsplit_test1())
#sqsplit_test2()
#sqsplit_test3()

class TreeNode(object):

    def __init__(self, left, right, feature, cut, prediction):
        self.left = left
        self.right = right
        self.feature = feature
        self.cut = cut
        self.prediction = prediction

#root2 = TreeNode(left_leaf, right_leaf, 0, 1 , 1.5)

def showVals(label,val1,val2):
    print(label)
    print(val1)
    print(val2)

def makeDecisionTree(xTr, yTr, node):

    n,D = xTr.shape

    #Start with a single node containing all points. Calculate mc and S.
    mc = np.sum(yTr) / len(yTr) #mc = the mean of yTr

    getSqrVar = lambda y, mc: (y-mc) ** 2

    S = np.sum(np.array([getSqrVar(y,mc) for y in yTr]))

    # See if the xTr consists of one, repeated feature vector.
    xTrHomogeneous = True
    firstxTrRow = xTr[0]
    for row in xTr:
        if not np.array_equal(firstxTrRow, row):
            xTrHomogeneous = False

    # If all the points in the node have the same value for all the independent variables, stop!
    if xTrHomogeneous:
        return TreeNode(None, None, None, None, mc)

    # If all data points in the data set share the same label we stop splitting and create a leaf with label LaTeX: yy
    if not hasDupe(yTr):
        return TreeNode(None, None, None, None, yTr[0])

    feature, cut, avgPrediction = sqsplit(xTr, yTr)

    # Generate the xTr and yTr sets for children, using the appropriate parameters:
    xTrLeft = np.empty((0, D), xTr.dtype)
    xTrRight = np.empty((0, D), xTr.dtype)
    yTrLeft = []
    yTrRight = []
    index = 0
    for row in xTr:
        if row[feature] <= cut:
            xTrLeft = np.vstack([xTrLeft,row])
            yTrLeft.append(yTr[index])
        else:
            xTrRight = np.vstack([xTrRight,row])
            yTrRight.append(yTr[index])
        index += 1

    showVals("xTr Left/Right",xTrLeft,xTrRight)

    yTrLeftNumpy = np.array(yTrLeft)
    yTrRightNumpy = np.array(yTrRight)

    if len(yTrLeftNumpy) == 0 or len(yTrRightNumpy) == 0:
        return TreeNode(None, None, None, None, avgPrediction)

    leftPred = np.sum(yTrLeftNumpy) / len(yTrLeftNumpy)
    rightPred = np.sum(yTrRightNumpy) / len(yTrRightNumpy)

    node.left = build(xTrLeft,yTrLeftNumpy, node)
    node.right = build(xTrRight,yTrRightNumpy, node)

    return TreeNode(node.left, node.right, feature, cut, avgPrediction)


# Now root2 is the tree we desired
def cart(xTr,yTr):

    #  Otherwise, search over all binary splits of all variables for the one which will reduce S as much as possible.
    return build(xTr,yTr,None)


#test case 1
def cart_test1():
    t=cart(xor4,yor4)
    return DFSxor(t)

#test case 2
def cart_test2():
    y = np.random.rand(16);
    t = cart(xor4,y);
    yTe = DFSpreds(t)[:];
    # Check that every label appears exactly once in the tree
    y.sort()
    yTe.sort()
    return np.all(np.isclose(y, yTe))

#test case 3
def cart_test3():
    xRep = np.concatenate([xor2, xor2])
    yRep = np.concatenate([yor2, 1-yor2])
    t = cart(xRep, yRep)
    return DFSxorUnsplittable(t)

#test case 4
def cart_test4():
    X = np.ones((5, 2)) # Create a dataset with identical examples
    y = np.ones(5)

    # On this dataset, your cart algorithm should return a single leaf
    # node with prediction equal to 1
    t = cart(X, y)

    # t has no children
    children_check = (t.left is None) and (t.right is None)

    # Make sure t does not cut any feature and at any value
    feature_check = (t.feature is None) and (t.cut is None)

    # Check t's prediction
    prediction_check = np.isclose(t.prediction, 1)
    return children_check and feature_check and prediction_check

#test case 5
def cart_test5():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])

    t = cart(X, y) # your cart algorithm should generate one split

    # check whether you set t.feature and t.cut to something
    return t.feature is not None and t.cut is not None

#t=cart(xor4,yor4)
#print("LET's DO ITTT")
#X = np.arange(4).reshape(-1, 1)
#print(X)
#y = np.array([0, 0, 1, 1])
#print(y)

#t = cart(X, y) # your cart algorithm should generate one split

print(hasDupe(y))
