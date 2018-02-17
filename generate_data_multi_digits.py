# import pacakges as necessary
from datasets import nDigits
from random import *
import numpy as np
import argparse
import os
import imutils



# Generate data from mnist data - each image contains 1,2 or 3 digits along with its y-lables
# image size 84,84
# y-lable = [len, digit-1,digit-2, digit-3]
# len = number of Digits
# digit = 10 if there is no digit
# NOTE: This script deletes the outfile if it exists

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output .npz file")
ap.add_argument("-t", "--train-size", type=int, default=50000,
	help="training set sample size")
ap.add_argument("-s", "--test-size", type=int, default=30000,
	help="test set sample size")
ap.add_argument("-i", "--image-size", type=int, default=0,
	help="-i 1 for half size image default is fullsize image")
ap.add_argument("-c","--compressed",type=int,default=1,
    help="compress? 0 is no-compression 1 is compression")
args = vars(ap.parse_args())

#intialise key variables
ntrain = args["train_size"]
ntest = args["test_size"]
fname = args["output"]
# frequency table for sample data - keys indicate number of digit, value indicate number of samples
freqTrain={'1':0,'2':0,'3':0}
freqTest={'1':0,'2':0,'3':0}
# array for generated data
X_train = []
y_train = []
X_test = []
y_test =[]

# check outfile - if exists then delete it
if (os.path.exists(fname)):
    os.remove(fname)

# generate training data
trainset = nDigits(train=True)
for i in range(ntrain):
	(X,y)=trainset.getImage("_2",randint(1,3))
	if args["image_size"]==1:
		X = imutils.resize(X,width=42)
	X_train.append(X)
	y_train.append(y)
	freqTrain[str(y[0])] += 1

X_train = np.array(X_train)
y_train = np.array(y_train)


# generate test data
trainset = nDigits(train=False)
for i in range(ntest):
	(X,y)=trainset.getImage("_2",randint(1,3))
	if args["image_size"]==1:
		X = imutils.resize(X,width=42)
	X_test.append(X)
	y_test.append(y)
	freqTrain[str(y[0])] += 1

X_test = np.array(X_test)
y_test = np.array(y_test)

if args["compressed"]==0:
    np.savez(fname,Xtrain=X_train, ytrain=y_train, Xtest=X_test,ytest=y_test)
else:
    np.savez_compressed(fname,Xtrain=X_train, ytrain=y_train, Xtest=X_test,ytest=y_test)

print("[INFO] data wrtitten to {}\n".format(fname))

print("*** Training data stats ****")
print("[INFO] X_train.shape {} y_train.shape {}".format(X_train.shape, y_train.shape))
print("[INFO] Frequecy table {}\n".format(freqTrain))
print("*** Test data stats ****")
print("[INFO] X_test.shape {} y_test.shape {}".format(X_test.shape, y_test.shape))
print("[INFO] Frequecy table {}".format(freqTest))
