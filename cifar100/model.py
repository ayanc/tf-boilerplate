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

        out = tf.nn.conv2d(inp,w,[1,stride,stride,1],padding)
        out = out + b

        if ifrelu:
            out = tf.nn.relu(out)

        return out

    # inp = tensor/variable/placeholder with input to network
    # self.out will be output of network (without softmax)
    def __init__(self,inp,phase=None,numc=3,numw=64):

        self.weights = {}
        self.phase = phase
                
        out = (tf.to_float(inp)-124.375)/68.4242

        # Data augmentation if training
        if phase is not None:
            out0 = tf.pad(out,[[0,0],[4,4],[4,4],[0,0]])
            out0 = tf.random_crop(out0,out.get_shape())
            out0 = tf.reshape(out0,[-1,32,3])
            out0 = tf.image.random_flip_left_right(out0)
            out0 = tf.reshape(out0,out.get_shape())

            out = tf.cond(tf.equal(phase,0),
                          lambda: out0, lambda: out)


        # Build conv-net    
        oscale = [1,2,4,8,8]
        prev = 3

        # 4 groups of numc Conv + 1 2x2 pooling layers
        for i in range(4):
            for j in range(numc):
                cur = int(oscale[i]*numw)
                out = self.conv(out,[3, prev, cur],
                                'conv'+str(i+1)+'_'+str(j+1))
                prev = cur
            out = tf.nn.max_pool(out,[1, 2, 2, 1],[1, 2, 2, 1],'VALID')

        # Final group of "fc" layers    
        sz=2
        i = 4
        for j in range(numc):
            cur = int(oscale[i]*numw)
            out = self.conv(out,[sz, prev, cur],
                            'conv'+str(i+1)+'_'+str(j+1),padding='VALID')
            sz=1
            prev = cur

            
        # Single fc-layer (implemented as 1x1 conv) at end to make predictions
        out = self.conv(out,[1, prev, 100],'fc1',
                        padding='VALID',ifrelu=False)
        out = tf.reshape(out,[-1, 100])
        self.out = out
        self.pred = tf.argmax(self.out,1)

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
