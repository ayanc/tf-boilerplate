# Ayan Chakrabarti <ayanc@ttic.edu>
## Parameters

# Batch size per iteration
BSZ=100
# Number of iterations per epoch
EPOCH=(45000/BSZ)
# Number of iterations to run
MAX_ITER=EPOCH*400
# Momentum
MOM=0.9
# Weight decay (1/2*wd |w|^2 for all non-bias weights)
WD=0.001
# Directory in which to store weights
WTS_DIR = 'wts/'

# How frequently to display
DISP_FREQ=100
# How frequently to validate
VAL_FREQ=1000
# How many batches in val set
VAL_ITER=5000//BSZ

# Frequency at which to checkpoint
SAVE_FREQ=10*EPOCH
# Subset of checkpoints to keep
KEEP_FREQ=0

# learning rate schedule
def get_lr(iter):
    base_lr = 0.02
    if iter < 200*EPOCH:
        lr = base_lr
    elif iter < 300*EPOCH:
        lr = base_lr/2.0
    elif iter < 350*EPOCH:
        lr = base_lr/4.0
    elif iter < 375*EPOCH:
        lr = base_lr/8.0
    else:
        lr= base_lr/16.0
    return lr
