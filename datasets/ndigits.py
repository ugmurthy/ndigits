# import packages as needed
from keras.datasets import mnist
import numpy as np
from random import *

class nDigits:
    def __init__(self,tstr="_1",train=True):
        # load mnist data
        (self._X,self._Y),(self._Xt,self._Yt) = mnist.load_data()

        # based on whether training or test data is needed set X,Y according
        if not train:
            self._X = self._Xt
            self._Y = self._Yt

        # keep a note of sample sizes
        self.nsamples = self._X.shape[0]

        self.v2=np.zeros((28,28),dtype='float32')
        # valid chars for template. update this string if u introduce a new template
        self.valid_tstr = 'IX_YL'  # to add V,Y,L
        # each string represent 'n' options for template - indicate that here
        self.valid_options = {'I':3,'X':2,'_':3,'Y':4,'L':4}
        # actual templates as an array of arrays indexed by template char and option number
        self.templates={
            'I': np.array([
            [[1,0,0],[2,0,0],[3,0,0]],
            [[0,1,0],[0,2,0],[0,3,0]],
            [[0,0,1],[0,0,2],[0,0,3]]
            ]),
            'X': np.array([
            [[1,0,0],[0,2,0],[0,0,3]],
            [[0,0,1],[0,2,0],[3,0,0]]
            ]),
            '_': np.array([
            [[1,2,3],[0,0,0],[0,0,0]],
            [[0,0,0],[1,2,3],[0,0,0]],
            [[0,0,0],[0,0,0],[1,2,3]]
            ]),
            'L': np.array([
            [[0,1,0],[0,2,3],[0,0,0]],
            [[0,1,0],[2,3,0],[0,0,0]],
            [[0,0,0],[1,2,0],[0,3,0]],
            [[0,0,0],[0,1,2],[0,3,0]],
            ]),
            'Y': np.array([
            [[0,1,0],[0,2,0],[0,0,3]],
            [[0,1,0],[0,2,0],[3,0,0]],
            [[1,0,0],[0,2,0],[0,3,0]],
            [[0,0,1],[0,2,0],[0,3,0]],
            ]),
            }

    # given a template string and number of Digits
    # getImage returns an array of images of MNIST digits along with an array of
    # y-labels for the corresponding images
    def getImage(self,tstr,nlen=3):
        # generate nlen digits randomly from MNIST data
        # arrange it using template
        template = self._decodeTemplate(tstr)
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

        # stack a 84,84 pixel image consisiting of
        # 1,2,or 3 digits
        digits = [self.v2, digits[0],digits[1],digits[2]]
        X=self._arrange(digits,template)
        Y=np.hstack(labels)
        return (X,Y)

    # generates an array of valid template strings
    def getTemplates(self):
        vlist = self.valid_options
        tlist = []
        for (k,i) in enumerate(vlist):
            for j in range(int(vlist[i])):
                tlist.append(i+str(j+1))
        return tlist

    # Helper functions
    def _arrange(self,digits,template):
        Rows = []
        for i in range(3):
            row = []
            for j in range(3):
                idx = template[i,j]
                #print(idx)
                row.append(digits[idx])
            # a row is now assembled as per getfromTemplate
            Rows.append(row)
        # all digits are ready for stacking
        return np.vstack([
            np.hstack(Rows[0]),
            np.hstack(Rows[1]),
            np.hstack(Rows[2])
        ])

    def _templateOK(self,template):
        # template should be 3,3
        # each row sum should be >0 <= 3
        # total matrix sum should be 6
        rsum=0
        tsum=template.sum()
        if not(tsum <= 6 and tsum > 0):
            return False
        if template.shape != (3,3):
            return False
        for i in range(3):
            rsum=template[i,:].sum()
            if not (rsum > 0 and rsum <= 3):
                return False
        return True

    def _decodeTemplate(self,tstr):

        if len(tstr)>= 2 :
            tstr = tstr[:2] # truncate to 2 chars
        else:
            return self._decodeTemplate('_2') # return this as default
        # split it
        try:
            idx = int(tstr[1])
        except:
            return self._decodeTemplate('_2') # return this as default

        tstr = tstr[0]

        # check if valid template char
        if tstr not in self.valid_tstr:
            return self._decodeTemplate('_2') # return this as default
        # check if idx is within the options we have
        if not (idx <= self.valid_options[tstr]):
            return self._decodeTemplate('_2') # return this as default

        idx -= 1 # ensure our index is in bounds
        # return template array - a numpy array
        return self.templates[tstr][idx]
