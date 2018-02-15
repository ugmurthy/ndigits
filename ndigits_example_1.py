from datasets import nDigits
import matplotlib.pyplot as plt
from random import *
import numpy as np

# This EXAMPLE will show :
#  1] How to create a image with 3 MNIST digits arranged in a pre-defined pattern
#  2] print a list of predefined pattern in the class nDigit.

## NOTE:
## using the class to generate single images is inefficient. This should be use used for
## generate large number of smaple of training and test data for later use

# [1]
A = nDigits()
tlist = A.getTemplates()

dlen=3 # number of digits needed : max is 3

template = tlist[randint(0,len(tlist))]
print("[INFO] Digits to extract {} from {} samples\n".format(dlen,A.nsamples))
print("[INFO] Template is {}\n".format(template))
# getImage and image label using template "X2" and consisting of dlen digits
(X,y)=A.getImage(template,dlen)

plt.imshow(X,cmap=plt.get_cmap('gray'))

plt.title("length = {0:d},digits = [{1:d},{2:d},{3:d}]".format(y[0],y[1],y[2],y[3]))
plt.show()

# [2]
# template is a 2 character string
# first char represents the arrangement of Digits
# second char represents one of the many options in that arrangement
# for example X has two arrangement and is represented by X1 and X2
# X1 is 3 digits(or less) arranged diagnally running from top left to bottom right
# X2 is 3 digits(or less) arranges diagnally running from top right to bottom left
# I is vertical columns - 3 options
# L is 'L' shaped arrangement - 4 options
# _ is horizontal arrangment - 3 options
# Y is half Y shaped arrangement - has 4 options
print("[INFO] List of templates :\n{}".format(A.getTemplates()))
