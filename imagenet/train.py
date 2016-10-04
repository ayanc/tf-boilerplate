# Ayan Chakrabarti <ayanc@ttic.edu>
import sys
import time
import tensorflow as tf
import numpy as np

import loader as ldr
import trainer as tr
import model as md
import utils as ut

# Config options here
LIST='data/train.txt'
WTS_DIR='wts/'
KEEPLAST = 2
SAVE_FREQ=1000

DISP_FREQ=10

BSZ=64
WEIGHT_DECAY=0.
LR = 0.001
MOM = 0.9

MAX_ITER = int(1.2e7)


# Check for saved weights
saved = ut.ckpter(WTS_DIR + 'iter_*.model.npz')
iter = saved.iter

# Set up batching
batcher = ut.batcher(LIST,BSZ,iter)

# Set up fetching and data prep
data = ldr.trainload(BSZ)
labels = tf.placeholder(shape=(BSZ,),dtype=tf.int32)

# Load model-def
net = md.model(data.batch,train=True)

# Load trainer-def
opt = tr.train(net,labels,LR,MOM,WEIGHT_DECAY)

# Start session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Load weights if any
if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest,sess)
    saved.clean(last=KEEPLAST)

# Training loop
s_loss = []
stop=False
try:
    # Load first batch
    imgs, nlbls = batcher.get_batch()
    _=sess.run(data.fetchOp,feed_dict=data.getfeed(imgs))
               
    while iter < MAX_ITER and not stop:

        # Swap in pre-fetched buffer into current start
        _=sess.run(data.swapOp)
        clbls = nlbls
        
        # Run training step    
        imgs,nlbls = batcher.get_batch()
        fdict = data.getfeed(imgs)
        fdict[labels] = clbls
        
        loss,_,_ = sess.run((opt.loss,opt.tstep,data.fetchOp),feed_dict=fdict)
        s_loss.append(loss)

        # Display frequently
        if iter % DISP_FREQ == 0:
            loss = np.mean(s_loss)
            s_loss = []
            tmstr = time.strftime("%Y-%m-%d %H:%M:%S")
            sys.stdout.write(tmstr + " [%09d] lr=%.2e Train loss = %.6f\n"
                             % (iter,LR,loss))
            sys.stdout.flush()

        iter=iter+1
                    
        # Save periodically
        if iter % SAVE_FREQ == 0:
            fname = WTS_DIR + "iter_%d.model.npz" % iter
            net.save(fname,sess)
            saved.clean(last=KEEPLAST)
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
    saved.clean(last=KEEPLAST)
    sys.stdout.write("Saved weights to " + fname + "\n")
    sys.stdout.flush()
