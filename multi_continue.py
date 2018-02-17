from keras.layers import Input, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import argparse
import time

t0=time.clock()
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default='multidigitnet.model',
	help="path to save model file. default is 'multidigitnet.model'")
ap.add_argument("-e", "--epoch", type=int, default=1,
	help="number of epochs to train default is 1")
ap.add_argument("-s", "--save",default='H.npz',
	help="path to save performance data. default is 'H.npz'")
ap.add_argument("-i", "--input",required=True,
	help="path to image file in .npz format'")
ap.add_argument("-r", "--train", type=int, default=10000,
	help="number of training sample to use - default is 10000")
ap.add_argument("-t", "--test", type=int, default=5000,
	help="number of test samples to use - default is 5000")
ap.add_argument("-d", "--dense", type=int, default=128,
	help="number of connections in Dense-1 - default is 128")
ap.add_argument("-l", "--lrate", type=float, default=0.01,
	help="learning rate - default is 0.01")
args = vars(ap.parse_args())
print("[INPUT PARAMETERS]")
for (i,k) in enumerate(args):
	print("\t{}\t{}".format(k,args[k]))
# tpyical usage
# python multi.py -m m4.model -s H4.npz -d 512 -l 0.01
# initialise filenames
mname=args['model']
hname=args['save']
fname=args['input']
testSamples=args['test']
trainSamples=args['train']
# get data

print("[INFO] Dense={} H={} M={} LR={}".format(args['dense'],args['save'],args['model'],args['lrate']))
print("[INFO] loading Training and Test data...")

nf=np.load(fname)
# parse it
X_train = nf['Xtrain']
X_test = nf['Xtest']
y_test = nf['ytest']
y_train = nf['ytrain']

# prune it to required sample sizes
X_train = X_train[0:trainSamples]
X_test = X_test[0:testSamples]
y_test = y_test[0:testSamples]
y_train = y_train[0:trainSamples]

# normalise to [0-1]
X_train = X_train/255.0
X_test = X_test/255.0

print("[INFO] preparing Training and test data...")
# convert y data to one-hot vector
# note y[0] is length, y[1..3] represent digits 1..3 - value of 10 means no digit present
# TRAIN DATA (y)
tr_len = to_categorical(y_train[:,0],num_classes=4) # len = number of digits
tr_d1 = to_categorical(y_train[:,1],num_classes=11) # digit-1
tr_d2 = to_categorical(y_train[:,2],num_classes=11) # digit-2
tr_d3 = to_categorical(y_train[:,3],num_classes=11) # digit-3

# TEST DATA (y)
te_len = to_categorical(y_test[:,0],num_classes=4)
te_d1 = to_categorical(y_test[:,1],num_classes=11)
te_d2 = to_categorical(y_test[:,2],num_classes=11)
te_d3 = to_categorical(y_test[:,3],num_classes=11)

# expand dimension of X DATA - i.e to add 'channel' dimension
X_train = np.expand_dims(X_train,3)
X_test = np.expand_dims(X_test,3)

print("[INFO] X, y shapes...")
print("\t Training data X.shape {}".format(X_train.shape))
print("\t Training data y.shapes len: {}, d1: {}, d2 {} d3: {}"
    .format(tr_len.shape,tr_d1.shape,tr_d2.shape,tr_d3.shape))
print("\n")
print("\t Test data X.shape {}".format(X_test.shape))
print("\t Test data y.shapes len: {}, d1: {}, d2 {} d3: {}"
    .format(te_len.shape,te_d1.shape,te_d2.shape,te_d3.shape))
print("\n")
t1=time.clock()
print("[INFO] Creating model...")
# create model
inputShape=X_train.shape[1:]
'''
inp = Input(shape=inputShape,name='image')
x=Conv2D(32,(3,3),padding='same',name='Conv1-3x3-32')(inp)
x=Activation('relu',name='relu-1')(x)
x=Conv2D(32,(3,3),padding='same',name='Conv2-3x3-32')(x)
x=Activation('relu',name='relu-2')(x)
x=MaxPooling2D(pool_size=(2,2),name='MaxPool-1')(x)
x=Dropout(0.25,name="Dropout-1")(x)
x=Conv2D(64,(3,3),padding='same',name='Conv3-3x3-64')(x)
x=Activation('relu',name='relu-3')(x)
x=Conv2D(64,(3,3),padding='same',name='Conv4-3x3-64')(x)
x=Activation('relu',name='relu-4')(x)
x=MaxPooling2D(pool_size=(2,2),name='MaxPool-2')(x)
x=Dropout(0.25,name='Droput-2')(x)
x=Flatten()(x)
bottleneck = Dense(args['dense'],activation='relu',name='Dense-1')(x)
nlen = Dense(4,activation='softmax',name='len')(bottleneck)
d1 = Dense(11,activation='softmax',name='digit-1')(bottleneck)
d2 = Dense(11,activation='softmax',name='digit-2')(bottleneck)
d3 = Dense(11,activation='softmax',name='digit-3')(bottleneck)
model = Model(inp,[nlen,d1,d2,d3])


print("[INFO] Compiling model...")
# compile the model
model.compile(
    loss={
    'len':'categorical_crossentropy',
    'digit-1':'categorical_crossentropy',
    'digit-2':'categorical_crossentropy',
    'digit-3':'categorical_crossentropy'
    },
    optimizer=SGD(lr=args['lrate']),
    metrics={
    'len':'accuracy',
    'digit-1':'accuracy',
    'digit-2':'accuracy',
    'digit-3':'accuracy'
    })
'''
model=load_model(mname)
print("[INFO] Training model...")
# train it
H=model.fit(X_train,[tr_len, tr_d1,tr_d2,tr_d3],
    validation_data=(X_test,[te_len,te_d1,te_d2,te_d3]),
    batch_size=128,
    epochs=args['epoch'],
    verbose=1)
t2=time.clock()

# save H
print("[INFO] Saving H data from training the network...")
np.savez(hname,loss=H.history['loss'],
    val_loss=H.history['val_loss'],
    len_loss=H.history['len_loss'],
    d1_loss=H.history['digit-1_loss'],
    d2_loss=H.history['digit-2_loss'],
    d3_loss=H.history['digit-3_loss'],
    len_acc=H.history['len_acc'],
    d1_acc=H.history['digit-1_acc'],
    d2_acc=H.history['digit-2_acc'],
    d3_acc=H.history['digit-3_acc'],
    val_len_loss=H.history['val_len_loss'],
    val_d1_loss=H.history['val_digit-1_loss'],
    val_d2_loss=H.history['val_digit-2_loss'],
    val_d3_loss=H.history['val_digit-3_loss'],
    val_len_acc=H.history['val_len_acc'],
    val_d1_acc=H.history['val_digit-1_acc'],
    val_d2_acc=H.history['val_digit-2_acc'],
    val_d3_acc=H.history['val_digit-3_acc'])

#for keys in H.history:
#    print("H.history['{}']".format(keys))


print("[INFO] Saving the trained model as '{}' for future use".format(mname))
model.save(mname)
print("[INFO] Time taken for data preparation : {}".format(t1-t0))
print("[INFO] Time taken for model prep and training: {}".format(t2-t1))
print("[INPUT PARAMETERS]")
for (i,k) in enumerate(args):
	print("\t{}\t{}".format(k,args[k]))
print("[INFO] InputShape {}".format(inputShape))
