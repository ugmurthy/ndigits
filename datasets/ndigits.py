# import packages as needed
from keras.datasets import mnist
import numpy as np
from random import *

class nDigits:
    def __init__(self,train=True):
        # load mnist data
        (self._X,self._Y),(self._Xt,self._Yt) = mnist.load_data()

        # based on whether training or test data is needed set X,Y according
        if not train:
            self._X = self._Xt
            self._Y = self._Yt

        # keep a note of sample sizes
        self.nsamples = self._X.shape[0]

        # keep black bands handy for stacking
        # black strip of 3 digits horizontally stacked
        self.v1=np.zeros((28,84),dtype='float32')
        # balck cell equivalent to 1 digit of 28x28 pixels
        self.v2=np.zeros((28,28),dtype='float32')


# getdigits ny default 3 digits
    def get(self,nlen=3):
        # generate nlen digits randomly from mnist data
        digits=[]
        labels=[]
        labels.append(nlen)
        for i  in range(nlen):
            idx=randint(0,self.nsamples-1)
            digits.append(self._X[idx])
            # convert one-hot-shot vector to number
            labels.append(self._Y[idx])

        if nlen<3:
            digits.append(self.v2)
            # label of 10 implies no digit
            labels.append(10)

        if nlen<2:
            digits.append(self.v2)
            # label of 10 implies no digit
            labels.append(10)

        # stack a 84,84 pixel image consisiting of 1,2,or 3 digits
        X=np.vstack((self.v1,np.hstack(digits),self.v1))
        Y=np.hstack(labels)
        return (X,Y)
