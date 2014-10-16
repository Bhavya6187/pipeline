import caffe
import numpy as np


input_image = caffe.io.load_image('test.png')
c = 'examples/cifar10/'
net = caffe.Classifier(c+'ba_deploy.protxt',c+'ba_iter_4000.caffemodel',raw_scale=255)
prediction = net.predict([input_image])
