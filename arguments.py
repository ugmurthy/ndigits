import argparse

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
