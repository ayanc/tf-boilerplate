#!/usr/bin/env python
# Ayan Chakrabarti <ayanc@ttic.edu>

import sys
import re
import numpy as np

import ntplot as nt

# Parse log files
def get_metric(lines,name,func):
    match = '\[(\d+)\].*' + name + '\s*=\s*([\d.-]+)'
    mset = {}
    for j in range(len(lines)):
        s = re.search(match,lines[j])
        if s is not None:
            mset[int(s.group(1))] = float(s.group(2))

    if len(mset.keys()) == 0:
        return None, None
    
    x = np.int64(sorted(mset.keys()))
    y = np.float64([func(mset[k]) for k in sorted(mset.keys())])

    return x,y

if len(sys.argv) != 3:
    sys.exit('Call as ' + sys.argv[0] + ' logfile outfile.html')

lines = open(sys.argv[1]).readlines()
fig = nt.figure()

x,y = get_metric(lines,'Train loss',lambda x: x)
if x is not None:
    fig.plot(x,y,'loss')

x,y = get_metric(lines,'Val accuracy',lambda x: 1.-x/100.)
if x is not None:
    fig.plot(x,y,'Val Err.')

fig.save(sys.argv[2])
