import numpy as np
import matplotlib.pyplot as plt

h10 = np.load('H10.npz')

xlen = h10['loss'].shape[0]
# plot the training loss and accuracy
linestyles = ['-', '--', '-.', ':']

plt.style.use("ggplot")
plt.figure()
# red '-'
plt.plot(np.arange(0,xlen), h10["loss"],linestyle=linestyles[2],color='red', label="train_loss")
plt.plot(np.arange(0,xlen), h10["len_loss"], label="len_loss")
plt.plot(np.arange(0,xlen), h10["d1_loss"], label="d1_loss")
plt.plot(np.arange(0,xlen), h10["d2_loss"], label="d2_loss")
plt.plot(np.arange(0,xlen), h10["d3_loss"], label="d3_loss")
# blue '--'
plt.plot(np.arange(0,xlen), h10["len_acc"],linestyle=linestyles[1],color='blue', label="len_acc")
plt.plot(np.arange(0,xlen), h10["d1_acc"], label="d1_acc")
plt.plot(np.arange(0,xlen), h10["d2_acc"], label="d2_acc")
plt.plot(np.arange(0,xlen), h10["d3_acc"], label="d3_acc")

# red '-.'
plt.plot(np.arange(0,xlen), h10["val_loss"], label="val_loss")
plt.plot(np.arange(0,xlen), h10["val_len_loss"], label="val_len_loss")
plt.plot(np.arange(0,xlen), h10["val_d1_loss"], label="val_d1_loss")
plt.plot(np.arange(0,xlen), h10["val_d2_loss"], label="val_d2_loss")
plt.plot(np.arange(0,xlen), h10["val_d3_loss"], label="val_d3_loss")
plt.plot(np.arange(0,xlen), h10["val_len_acc"], label="val_len_acc")
plt.plot(np.arange(0,xlen), h10["val_d1_acc"], label="val_d1_acc")
plt.plot(np.arange(0,xlen), h10["val_d2_acc"], label="val_d2_acc")
plt.plot(np.arange(0,xlen), h10["val_d3_acc"], label="val_d3_acc")
# blue ':'
plt.title("Training loss and accuracy - multidigits")
plt.ylabel("Loss/Accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()
