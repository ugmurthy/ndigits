from datasets import nDigits
import matplotlib.pyplot as plt
from random import *

A = nDigits()


dlen=randint(1,3)
print("[INFO] Digits to extract {} from {} samples\n".format(dlen,A.nsamples))

(X,y)=A.get(dlen)

plt.imshow(X,cmap=plt.get_cmap('gray'))
#plt.title("length = {} digits = {}".format(y[0],y[1:]))
plt.title("length = {0:d},digits = [{1:d},{2:d},{3:d}]".format(y[0],y[1],y[2],y[3]))
plt.show()
