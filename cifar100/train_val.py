# Ayan Chakrabarti <ayanc@ttic.edu>
import sys
import time
import tensorflow as tf
import numpy as np

import cifar100 as cf
import trainer as tr
import model as md
import utils as ut

from params import *


# Load dataset        
tset,vset = cf.get_tval()

# Restore any saved weights
saved = ut.ckpter(WTS_DIR + 'iter_*.model.npz')
iter = saved.iter

# Set up batching    
tbatch = ut.batcher(tset[0],tset[1],BSZ,iter)
vbatch = ut.batcher(vset[0],vset[1],BSZ,0,False)

# Create placeholders
data = tf.placeholder(shape=(BSZ,)+tset[0].shape[1:],
                      dtype=tset[0].dtype)
labels = tf.placeholder(shape=(BSZ,)+tset[1].shape[1:],
                        dtype=tset[1].dtype)

lr = tf.placeholder(shape=(),dtype=tf.float32)
phase = tf.placeholder(shape=(),dtype=tf.int32)

# Load model-def
net = md.model(data,phase)

# Load trainer-def
opt = tr.train(net,labels,lr,MOM,WD)

# Start session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Load weights if any
if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest,sess)
    saved.clean(KEEP_FREQ)

# Training loop
s_loss = []
stop=False
try:
    while iter < MAX_ITER and not stop:

        # Validate periodically
        if iter % VAL_FREQ == 0:
            acc = 0.0
            for i in range(VAL_ITER):
                d,l = vbatch.get_batch()
                p = sess.run(net.pred, feed_dict = {data: d, phase: 1})
                acc = acc + np.mean(np.float64(p==l))
            acc = acc / float(VAL_ITER) * 100.0
            tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
            sys.stdout.write(tmstr + " [%08d] Val accuracy = %.2f\n"
                             % (iter,acc))
            sys.stdout.flush()

        # Run training step    
        d,l = tbatch.get_batch()
        it_lr = get_lr(iter)
        fdict = {data: d, labels: l, lr: it_lr, phase: 0}
        loss,_ = sess.run((opt.loss,opt.tstep),feed_dict=fdict)
        s_loss.append(loss)

        # Display frequently
        if iter % DISP_FREQ == 0:
            loss = np.mean(s_loss)
            s_loss = []
            tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
            sys.stdout.write(tmstr + " [%08d] lr=%.2e Train loss = %.6f\n"
                             % (iter,it_lr,loss))
            sys.stdout.flush()

        iter=iter+1
                    
        # Save periodically
        if iter % SAVE_FREQ == 0:
            fname = WTS_DIR + "iter_%d.model.npz" % iter
            net.save(fname,sess)
            saved.clean(KEEP_FREQ)
            sys.stdout.write("Saved weights to " + fname + "\n")
            sys.stdout.flush()

except KeyboardInterrupt: # Catch ctrl+c
    sys.stderr.write("Stopped!\n")
    sys.stderr.flush()
    stop = True
    pass

if saved.iter < iter:    
    fname = WTS_DIR + "iter_%d.model.npz" % iter
    net.save(fname,sess)
    saved.clean(KEEP_FREQ)
    sys.stdout.write("Saved weights to " + fname + "\n")
    sys.stdout.flush()
