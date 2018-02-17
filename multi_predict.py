from keras.layers import Input, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import to_categorical
from datasets import nDigits
import numpy as np
import imutils
from random import *
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to existing model")
ap.add_argument("-n", "--num", type=int, default=32,
	help="number of samples to use for prediction")

args = vars(ap.parse_args())

# tpyical usage
# python multi.py -m m4.model -s H4.npz -d 512 -l 0.01
# initialise filenames
mname=args['model']
fname="multidigithalf_c.npz"
freq={'1':0,'2':0,'3':0}
X_=[]
y_=[]

predSamples=args['num']
# get data
print("[INFO] Generating Sample data for prediction...")
sampleSet = nDigits(train=True)
for i in range(predSamples):
	(X,y)=sampleSet.getImage("_2",randint(1,3))
	X = imutils.resize(X,width=42)
	X_.append(X)
	y_.append(y)
	freq[str(y[0])] += 1

X = np.array(X_)
y = np.array(y_)


# normalise to [0-1]
X = X/255.0
Z = X # keep a copy
# expand dimension of X DATA
X = np.expand_dims(X,3)



print("[INFO] Sample data X.shape {}".format(X.shape))
print("[INFO] loading model...")
model=load_model(mname)

prediction = model.predict(X,batch_size=32)

# get values for each output
len = prediction[0].argmax(axis=1)
d1 = prediction[1].argmax(axis=1)
d2 = prediction[2].argmax(axis=1)
d3 = prediction[3].argmax(axis=1)

# convert to a format so that we can compare with y
prediction = np.transpose(np.vstack([len,d1,d2,d3]))

# compare with y
tlen = prediction[:,0]==y[:,0]
t1 = prediction[:,1]==y[:,1]
t2 = prediction[:,2]==y[:,2]
t3 = prediction[:,3]==y[:,3]

# result of comparision each element will be a boolean True if prediction = y
result = np.transpose(np.vstack([tlen,t1,t2,t3]))
print(result)

# check result - if all value in a row are true then we have predicted correctly else incorrect
correct = 0
for i in range(result.shape[0]):
	if result[i,:].sum() == 4: # all 4 values if True adds upto 4
		correct += 1.0
print("[INFO] correct predictions = {}%".format(100.0*correct/result.shape[0]))


# plot the 4 images with y data as subplot title
ax=plt.subplot(221)
ax.imshow(Z[1],cmap=plt.get_cmap("gray"))
ax.set_title("len={} digits:{}".format(prediction[1][0],prediction[1][1:]))
ax.axis('off')

bx=plt.subplot(222)
bx.imshow(Z[2],cmap=plt.get_cmap("gray"))
bx.set_title("len={} digits:{}".format(prediction[2][0],prediction[2][1:]))
bx.axis('off')

cx=plt.subplot(223)
cx.imshow(Z[3],cmap=plt.get_cmap("gray"))
cx.set_title("len={} digits:{}".format(prediction[3][0],prediction[3][1:]))
cx.axis('off')

dx=plt.subplot(224)
dx.imshow(Z[4],cmap=plt.get_cmap("gray"))
dx.set_title("len={} digits:{}".format(prediction[4][0],prediction[4][1:]))
dx.axis('off')
plt.suptitle("Predictions (10 means no digit)")
plt.show()
