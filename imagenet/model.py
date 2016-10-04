# Ayan Chakrabarti <ayanc@ttic.edu>
import tensorflow as tf
import numpy as np

class model:
    def conv(self,inp,ksz,name,stride=1,padding='SAME',ifrelu=True):
        ksz = [ksz[0],ksz[0],ksz[1],ksz[2]]

        # xavier init
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        w = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        self.weights[name+'_W'] = w

        # constant init
        b = tf.Variable(tf.constant(0,shape=[ksz[3]],dtype=tf.float32))
        self.weights[name+'_b'] = b

        out = tf.nn.conv2d(inp,w,[stride,stride,1,1],padding)
        out = out + b

        if ifrelu:
            out = tf.nn.relu(out)

        return out

    # Implement VGG-16 architecture
    #   inp = tensor/variable/placeholder with input to network
    #   self.out will be output of network (without softmax)
    def __init__(self,inp,train=False):

        self.weights = {}

        imean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3])
        out = (inp-imean)
        prev=3

        numc = [2,2,3,3,3]
        numw = [64,128,256,512,512]
        
        for i in range(5):
            for j in range(numc[i]):
                cur = int(numw[i])
                out = self.conv(out,[3, prev, cur],
                                'conv'+str(i+1)+'_'+str(j+1))
                prev = cur
            out = tf.nn.max_pool(out,[1, 2, 2, 1],[1, 2, 2, 1],'VALID')

        out = self.conv(out,[7, prev, 4096],'fc6',padding='VALID')
        if train:
            out = tf.nn.dropout(out,0.5)
        out = self.conv(out,[1, 4096, 4096],'fc7',padding='VALID')
        if train:
            out = tf.nn.dropout(out,0.5)
        out = self.conv(out,[1, 4096, 1000],'fc8',padding='VALID',ifrelu=False)

        if train:
            self.out = tf.reshape(out,[-1, 1000])            
        else:
            self.out = tf.reduce_mean(out,[1, 2])

    # Load weights from an npz file
    def load(self,fname,sess):
        wts = np.load(fname)
        for k in wts.keys():
            wvar = self.weights[k]
            wk = wts[k].reshape(wvar.get_shape())
            sess.run(wvar.assign(wk))

    # Save weights to an npz file
    def save(self,fname,sess):
        wts = {}
        for k in self.weights.keys():
            wts[k] = self.weights[k].eval(sess)
        np.savez(fname,**wts)
