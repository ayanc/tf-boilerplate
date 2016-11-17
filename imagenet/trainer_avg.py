# Ayan Chakrabarti <ayanc@ttic.edu>

# This is a version of trainer that supports averaging gradients
# across multiple batches. See train_avg.py for usage.

import tensorflow as tf
import numpy as np

class train:
    def __init__(self,model,labels,lr,mom,wd):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                model.out,labels))

        if wd > 0.:
            # Define L2 weight-decay on all non-bias vars
            reg = list()
            for k in model.weights.keys():
                wt = model.weights[k]
                if len(wt.get_shape()) > 1:
                    reg.append(tf.nn.l2_loss(wt))
                    self.reg = tf.add_n(reg)

                    # This is our minimization objective
                    self.obj = self.loss + wd*self.reg
        else:
            self.obj = self.loss
            
        # Set up momentum trainer
        self.opt = tf.train.MomentumOptimizer(lr,mom)

        # Setup optimizer to double gradients of biases, and to
        # average gradients across iterations before applying them.
        gv1 = self.opt.compute_gradients(self.obj)

        # First create gradient variables
        gvars = list()
        for gv in gv1:
            gvars.append(tf.Variable(tf.zeros_initializer(
                gv[0].get_shape(),dtype=tf.float32),trainable=False))

        # Zero out variables before aggregating    
        self.grad0 = [v.initializer for v in gvars]

        # Add current gradient to aggregate
        self.gstep = [gvars[i].assign_add(gv1[i][0]).op
                      for i in range(len(gvars))]

        
        gv2 = list()
        for i in range(len(gv1)):
            if len(gv1[i][1].get_shape()) > 1:
                gv2.append((0.+gvars[i],gv1[i][1]))
            else:
                gv2.append((2.*gvars[i],gv1[i][1]))
        self.tstep = self.opt.apply_gradients(gv2)
