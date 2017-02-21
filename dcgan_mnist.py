"""
SUMMARY:  DCGAN, can generate real digits after 10 epochs. 
Ref:      [1] https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan.py
AUTHOR:   Qiuqiang Kong
Created:  2017.02.19
Modified: -
--------------------------------------
"""
import os
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat.models import Sequential, Model
from hat.layers.core import *
from hat.layers.cnn import *
from hat.layers.pooling import *
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat.layers.normalization import BN
import prepare_data as pp_data
import numpy as np
from PIL import Image


def train():
    # load data
    batch_size = 128
    tr_X, tr_y, va_X, va_y, te_X, te_y = pp_data.load_data()
    n_batches = int(tr_X.shape[0]/batch_size)
    
    # normalize data between [-1,1]
    tr_X = (tr_X-0.5) * 2
    tr_X = tr_X.reshape((50000,1,28,28))
    print tr_X.shape
    
    # generator
    a0 = InputLayer(100)
    a1 = Dense(128*7*7,act='linear')(a0)
    a1 = BN(axis=0)(a1)
    a1 = Reshape(out_shape=(128, 7, 7))(a1)
    a1 = Convolution2D(64,5,5,act='linear',border_mode=(2,2))(a1)
    a1 = BN(axis=(0,2,3))(a1)
    a1 = Activation('leaky_relu')(a1)
    a1 = UpSampling2D(size=(2,2))(a1)
    a1 = Convolution2D(32,5,5,act='linear',border_mode=(2,2))(a1)
    a1 = BN(axis=(0,2,3))(a1)
    a1 = Activation('leaky_relu')(a1)
    a1 = UpSampling2D(size=(2,2))(a1)
    a8 = Convolution2D(1,5,5,act='tanh',border_mode=(2,2), name='a8')(a1)
    
    g = Model([a0], [a8])
    g.compile()
    g.summary()
    
    # discriminator
    b0 = InputLayer((1,28,28), name='b0')
    b1 = Convolution2D(64,5,5,act='relu',border_mode=(0,0), name='b1')(b0)
    b1 = MaxPooling2D(pool_size=(2,2))(b1)
    b1 = Convolution2D(128,5,5,act='relu',border_mode=(0,0))(b1)
    b1 = MaxPooling2D(pool_size=(2,2))(b1)
    b1 = Flatten()(b1)
    b8 = Dense(1, act='sigmoid')(b1)
    d = Model([b0], [b8])
    d.compile()
    d.summary()
    
    # discriminator on generator
    d_on_g = Model()
    d.set_trainability(False)
    d_on_g.add_models([g, d])
    d.set_trainability(True)
    d_on_g.joint_models('a8', 'b0')
    d_on_g.compile()
    d_on_g.summary()
    
    # optimizer
    opt_d = Adam(1e-4)
    opt_g = Adam(1e-4)
    
    # optimization function
    f_train_d = d.get_optimization_func(target_dims=[2], loss_func='binary_crossentropy', optimizer=opt_d, clip=None)
    f_train_g = d_on_g.get_optimization_func(target_dims=[2], loss_func='binary_crossentropy', optimizer=opt_g, clip=None)
    
    noise = np.zeros((batch_size, 100))
    for epoch in range(100):
        print epoch
        for index in range(n_batches):
            # concatenate generated img and real image to train discriminator. 
            noise = np.random.uniform(-1, 1, (batch_size,100))
            batch_x = tr_X[index*batch_size:(index+1)*batch_size]
            batch_gx = g.predict(noise)
            batch_x_all = np.concatenate((batch_x, batch_gx))
            
            # assign real img label as 1, generated img label as 0
            batch_y_all = np.array([1] * batch_size + [0] * batch_size)     
            batch_y_all = batch_y_all.reshape((batch_y_all.shape[0],1))
            
            # save out generated img
            if index % 50 == 0:
                image = pp_data.combine_images(batch_gx)
                image = image*127.5+127.5
                if not os.path.exists("img_dcgan"): os.makedirs("img_dcgan")
                Image.fromarray(image.astype(np.uint8)).save(
                    "img_dcgan/" + str(epoch)+"_"+str(index)+".png")
 
            # train discriminator
            d_loss = d.train_on_batch(f_train_d, batch_x_all, batch_y_all)

            # assign generate img label as 1, so as to deceive discriminator
            noise = np.random.uniform(-1, 1, (batch_size,100))
            batch_y_all = np.array([1] * batch_size)
            batch_y_all = batch_y_all.reshape((batch_y_all.shape[0],1))
            
            # train generator
            g_loss = d_on_g.train_on_batch(f_train_g, noise, batch_y_all)
            print index, "d_loss:", d_loss, "\tg_loss:", g_loss

if __name__ == "__main__":
    train()
