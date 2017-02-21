'''
SUMMARY:  Download mnist dataset
Ref:      http://deeplearning.net/tutorial/code/logistic_sgd.py
AUTHOR:   Qiuqiang Kong
Created:  2016.05.01
Modified: 2016.07.19 Modify dataset path
--------------------------------------
'''
import cPickle
import gzip
import os
import numpy as np
import math

def load_data():
    dataset = 'mnist.pkl.gz'
    if not os.path.isfile(dataset):
        from six.moves import urllib
        print 'downloading data ... (16.2 Mb)'
        urllib.request.urlretrieve( 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset )
        
    f = gzip.open( dataset, 'rb' )
    train_set, valid_set, test_set = cPickle.load(f)
    [tr_X, tr_y] = train_set
    [va_X, va_y] = valid_set
    [te_X, te_y] = test_set
    f.close()
    return tr_X, tr_y, va_X, va_y, te_X, te_y

# Combine many images to one big image to save out    
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image