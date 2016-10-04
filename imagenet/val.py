# Ayan Chakrabarti <ayanc@ttic.edu>
import sys
import time
import tensorflow as tf
import numpy as np

import loader as ldr
import trainer as tr
import model as md
import utils as ut


VAL='data/val.txt'
SCALES=[256,512]


if len(sys.argv) != 2:
    exit('Call with path to model.npz file.')
    

imgs, lbls = ut.load(VAL)

# Create placeholders
data = tf.placeholder(tf.string)
size = tf.placeholder(tf.float32)

# Load model-def
net = md.model(ldr.testload(data,size))

# Start session
sess = tf.Session()
sys.stdout.write("Restoring from " + sys.argv[1] + "\n")
sys.stdout.flush()
net.load(sys.argv[1],sess)

pos = 0
total = 0
# Do this class-wise
for c in range(np.max(lbls)):
    idx = ut.find(lbls,c)
    total = total + len(idx)
    count = 0
    
    t0 = time.time()
    for i in range(len(idx)):
        img = imgs[idx[i]]
        score = 0.
        for s in range(len(SCALES)):
            score = score + sess.run(net.out,
                                     feed_dict={data: img, size: SCALES[s]})

        # Compute if in top-5
        score = np.fliplr(np.argsort(score))[:,0:5]
        score = np.sum([np.any(q==c) for q in score],dtype=np.int32)
        count = count + score

    t1 = time.time()
    pos = pos + count

    sys.stdout.write("\nClass " + str(c) + ": " + "%.2f" %
                     (np.float32(count)/np.float32(len(idx))*100.) +" %\n")
    sys.stdout.write("Time: " + ("%.2f" % (t1 - t0)) + "s.\n")
    sys.stdout.write("Running Total: " + "%.2f" %
                     (np.float32(pos)/np.float32(total)*100.) +" %\n")
    sys.stdout.flush()


sys.stdout.write("Total: " + "%.2f" %
                 (np.float32(pos)/np.float32(total)*100.) +" %\n")
sys.stdout.write("Pos: " + str(pos) + " Tot: " + str(total) + "\n")

    
