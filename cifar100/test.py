# Ayan Chakrabarti <ayanc@ttic.edu>
import sys
import tensorflow as tf
import numpy as np

import cifar100 as cf
import trainer as tr
import model as md
import utils as ut

BSZ=500

testset = cf.get_test()
# Set up batching    
tbatch = ut.batcher(testset[0],testset[1],BSZ,0,False)

# Create placeholders
data = tf.placeholder(shape=(BSZ,)+testset[0].shape[1:],
                      dtype=testset[0].dtype)

# Load model-def
net = md.model(data)

# Start session
sess = tf.Session()
sys.stdout.write("Restoring from " + sys.argv[1] + "\n")
sys.stdout.flush()
net.load(sys.argv[1],sess)

# Test
nIter = testset[0].shape[0] // BSZ
acc = 0.0
for i in range(nIter):
    d,l = tbatch.get_batch()
    p = sess.run(net.pred, feed_dict = {data: d})
    acc = acc + np.mean(np.float64(p==l))
acc = acc / float(nIter) * 100.0
sys.stdout.write("Test accuracy = %.2f\n" % acc)
sys.stdout.flush()
