# Ayan Chakrabarti <ayanc@ttic.edu>

# This version of the training script allows you to achieve an
# effective batch size (BSZ below) larger than what could fit in GPU
# memory. Select AVG_IT to be a factor of BSZ, such that BSZ/AVG_IT
# fits on your GPU.
#
# All iteration accounting (displayed, checkpoint name, etc.) is with
# respect to 'true batch size'.

import sys
import time
import tensorflow as tf
import numpy as np

import loader as ldr
import trainer_avg as tr
import model as md
import utils as ut

# Config options here
LIST='data/train.txt'
WTS_DIR='wts/'
KEEPLAST = 2
SAVE_FREQ=1000

DISP_FREQ=10

BSZ=128
AVG_IT = 2 # Divide batch over these many iterations
           # to fit in GPU memory. BSZ % AVG_IT must
           # be 0.

WEIGHT_DECAY=0.
LR = 0.01
MOM = 0.9

MAX_ITER = int(1.2e7)

# Update BSZ, lr
BSZ = BSZ // AVG_IT
LR_E = LR / float(AVG_IT)


# Check for saved weights
saved = ut.ckpter(WTS_DIR + 'iter_*.model.npz')
iter = saved.iter

# Set up batching
batcher = ut.batcher(LIST,BSZ,iter*AVG_IT)

# Set up data prep
data = ldr.trainload(BSZ)
labels = tf.placeholder(shape=(BSZ,),dtype=tf.int32)

# Load model-def
net = md.model(data.batch,train=True)

# Load trainer-def
opt = tr.train(net,labels,LR_E,MOM,WEIGHT_DECAY)

# Start session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Load saved weights if any
if saved.latest is not None:
    sys.stdout.write("Restoring from " + saved.latest + "\n")
    sys.stdout.flush()
    net.load(saved.latest,sess)
    saved.clean(last=KEEPLAST)

# Training loop
# Does clean shutdown on ctrl+c (saving weights)
# You can exploit this to easily put a time-limit (for cluster jobs)
# using the unix timeout command with the SIGINT signal. E.g:
# timeout --foreground -s INT 3.7h python ./train.py
s_loss = []
stop=False
try:
    # Load first batch
    imgs, nlbls = batcher.get_batch()
    _=sess.run(data.fetchOp,feed_dict=data.getfeed(imgs))
               
    while iter < MAX_ITER and not stop:

        # Swap in pre-fetched buffer into current input
        _=sess.run(data.swapOp)
        clbls = nlbls
        
        # Run training step & getch for next batch
        imgs,nlbls = batcher.get_batch()
        fdict = data.getfeed(imgs)
        fdict[labels] = clbls

        sess.run(opt.grad0)
        for j in range(AVG_IT):
            outs = sess.run([opt.loss,data.fetchOp]+
                            opt.gstep,feed_dict=fdict)
            s_loss.append(outs[0])
        sess.run(opt.tstep)
            
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
