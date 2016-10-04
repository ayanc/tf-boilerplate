# Ayan Chakrabarti <ayanc@ttic.edu>
import numpy as np
from os import getenv
import cPickle as cp

# Set an environment variable CIFAR100
# to point to the directory where you
# extracted the dataset.
_base=getenv('CIFAR100')

def load(name):
    d = cp.load(open(name,'rb'))
    labels = np.int32(d['fine_labels'])
    data = d['data'].reshape([-1,32,32,3])
    return data,labels

def get_test():
    data,labels = load(_base + '/test')
    return [data, labels]

def get_tval():
    data, labels = load(_base + '/train')

    tdata = data[:45000,...]
    tlbl = labels[:45000]

    vdata = data[45000:,...]
    vlbl = labels[45000:]

    return [tdata,tlbl],[vdata,vlbl]
