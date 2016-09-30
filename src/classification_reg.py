#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/Users/Bob/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()

model_def = caffe_root + 'examples/yearbook/deploy_regression.prototxt'
model_weights = caffe_root + 'examples/yearbook/VGG_Regre_iter_5882.caffemodel'
binary_proto_mean = caffe_root + 'data/yearbook/yearbook_mean.binaryproto'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( binary_proto_mean  , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )[0]

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# # load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mean_arr.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print 'mean-subtracted values:', zip('BGR', mu)

# # create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,1,0))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          171, 186)  # image size is 227x227

def evaluate(im_dir):
	image = caffe.io.load_image(im_dir)
                                #caffe_root + 'data/yearbook/' + im_dir
	transformed_image = transformer.preprocess('data', image)
	plt.imshow(image)

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()

	output_prob = output['fc8'][0]  # the output probability vector for the first image in the batch
	return output_prob +1905
	#year = output_prob.argmax()+ 1905
	#return year

def main():
	im_dir = '/Users/Bob/caffe/CS395T/CS395T/data/yearbook/M/2002_California_San-Diego_Mission-Bay_f-136.png'
	output_prob = evaluate(im_dir)
	print 'predicted class is:', output_prob
if __name__ == "__main__":main()













