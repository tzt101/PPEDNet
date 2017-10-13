import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import time
from sklearn.preprocessing import normalize
caffe_root = './PPEDNet/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):
	start = time.time()
	net.forward()
	end = time.time()
	print '%30s' % 'Executed PPEDNet in ', str((end - start)*1000), 'ms'
	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	accuracy = net.blobs['accuracy'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()
	rgb = ind.copy()
	rgb_gt = label.copy()

	pre_0 = [0,0,0]
	pre_1 = [1,1,1]
	pre_2 = [2,2,2]
	pre_3 = [3,3,3]
	pre_4 = [4,4,4]
	pre_5 = [5,5,5]
	pre_6 = [6,6,6]
	pre_7 = [7,7,7]
	pre_8 = [8,8,8]
	pre_9 = [9,9,9]
	pre_10 = [10,10,10]
	pre_11 = [11,11,11]
	color = np.array([pre_0,pre_1,pre_2,pre_3,pre_4,pre_5,pre_6,pre_7,pre_8,pre_9,pre_10,pre_11])
	for l in range(0,11): 
		r[ind==l] = color[l,0]
		g[ind==l] = color[l,1]
		b[ind==l] = color[l,2]
		r_gt[label==l] = color[l,0]
		g_gt[label==l] = color[l,1]
		b_gt[label==l] = color[l,2]
	
	rgb = np.zeros((ind.shape[0],ind.shape[1],3))
	rgb[:,:,0] = r
	rgb[:,:,1] = g
	rgb[:,:,2] = b	
	rgb_gt = np.zeros((ind.shape[0],ind.shape[1],3))
	rgb_gt[:,:,0] = r_gt
	rgb_gt[:,:,1] = g_gt
	rgb_gt[:,:,2] = b_gt	
	
	scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save('./PPEDNet/Scripts/predictions/' + str(i) + '_pr.png')
	scipy.misc.toimage(rgb_gt, cmin=0.0, cmax=255).save('./PPEDNet/Scripts/gt/' + str(i) + '_gt.png')



print 'Success!'

