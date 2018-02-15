from datasets import nDigits
import matplotlib.pyplot as plt
from random import *
import os
import numpy as np

### part 1 ###
# create one image consisting of 3 digits and show it
# this is to test/dmonstrate the use of nDigits class
A = nDigits()
#dlen=randint(1,3)
dlen=3
template='X2'
print("[INFO] Digits to extract {} from {} samples\n".format(dlen,A.nsamples))
# getImage and image label using template "X2" and consisting of dlen digits
(X,y)=A.getImage(template,dlen)

plt.imshow(X,cmap=plt.get_cmap('gray'))

plt.title("length = {0:d},digits = [{1:d},{2:d},{3:d}]".format(y[0],y[1],y[2],y[3]))
plt.show()
### end of Part 1 ###

### part 2 - write 2000 samples of training date and 1000 samples of test data to file

# frequency table for sample data - keys indicate number of digit, value indicate number of samples
freqTrain={'1':0,'2':0,'3':0}
freqTest={'1':0,'2':0,'3':0}

fname = '_test_.npz'
if (os.path.exists(fname)):
    os.remove(fname)
ntrain = 20
ntest = 10
X_train = []
y_train = []
X_test = []
y_test =[]

# generate training data
trainset = nDigits(train=True)
for i in range(ntrain):
    (X,y)=trainset.getImage("L3",nlen=randint(1,3))
    X_train.append(X)
    y_train.append(y)
    freqTrain[str(y[0])] += 1

X_train = np.array(X_train)
y_train = np.array(y_train)


# generate test data
testset = nDigits(train=False)
for i in range(ntest):
    (X,y)=testset.getImage("_3",nlen=randint(1,3))
    X_test.append(X)
    y_test.append(y)
    freqTest[str(y[0])] += 1

X_test = np.array(X_test)
y_test = np.array(y_test)

np.savez(fname,Xtrain=X_train, ytrain=y_train, Xtest=X_test,ytest=y_test)
print("[INFO] data wrtitten to {}".format(fname))

# plot 4 samples 2 test and 2 train


print("*** Training data stats ****")
print("[INFO] X_train.shape {} y_train.shape {}".format(X_train.shape, y_train.shape))
print("[INFO] Frequecy table {}\n".format(freqTrain))
print("*** Test data stats ****")
print("[INFO] X_test.shape {} y_test.shape {}".format(X_test.shape, y_test.shape))
print("[INFO] Frequecy table {}".format(freqTest))

# PART 3 : Load data from .npz file and display a few samples
# load npz file
nf = np.load(fname)
X_train = nf['Xtrain']
y_train = nf['ytrain']
X_test  = nf['Xtest']
y_test  = nf['ytest']

# get two random training data samples X,y
idx=randint(0,ntrain-1)
X1 = X_train[idx]
y1 = y_train[idx]

idx=randint(0,ntrain-1)
X2 = X_train[idx]
y2 = y_train[idx]

# get two random test data samples X,y
idx=randint(0,ntest-1)
X3 = X_test[idx]
y3 = y_test[idx]

idx=randint(0,ntest-1)
X4 = X_test[idx]
y4 = y_test[idx]

# plot the 4 images with y data as subplot title
ax=plt.subplot(221)
ax.imshow(X1,cmap=plt.get_cmap("gray"))
ax.set_title("Train:len={} digits:{}".format(y1[0],y1[1:]))
ax.axis('off')

bx=plt.subplot(222)
bx.imshow(X2,cmap=plt.get_cmap("gray"))
bx.set_title("Train:len={} digits:{}".format(y2[0],y2[1:]))
bx.axis('off')

cx=plt.subplot(223)
cx.imshow(X3,cmap=plt.get_cmap("gray"))
cx.set_title("Test:len={} digits:{}".format(y3[0],y3[1:]))
cx.axis('off')

dx=plt.subplot(224)
dx.imshow(X4,cmap=plt.get_cmap("gray"))
dx.set_title("Test:len={} digits:{}".format(y4[0],y4[1:]))
dx.axis('off')

plt.show()
