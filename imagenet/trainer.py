# Ayan Chakrabarti <ayanc@ttic.edu>
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

        # Setup optimizer to double gradients of biases You can add
        # other conditions here for more exotic settings, like
        # gradient clipping. For arbitrary learning rate multipliers,
        # I suggest creating a dict() in keyed by variable.name and
        # checking that below.
        gv1 = self.opt.compute_gradients(self.obj)
        gv2 = list()
        for gv in gv1:
            if len(gv[1].get_shape()) > 1:
                gv2.append(gv)
            else:
                gv2.append((2.*gv[0],gv[1]))
        self.tstep = self.opt.apply_gradients(gv2)
