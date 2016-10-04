# Tensorflow Boiler-plate
Ayan Chakrabarti <<ayanc@ttic.edu>>

This repository contains *boiler-plate* code that you can use as a
starting point to write your tensorflow code for neural-network
training. This isn't a library of classes for general purpose use:
modify the code to adapt it to your task. I have created this repo
mainly as a place to put reference code for students I work with. But
all code is released for public use under the
[MIT license](LICENSE.md).

There are currently two directories: 

1. `cifar100`: Somewhat simpler of the two setups. See `train_val.py`
   and `test.py` and follow pointers from there. If you want to run a
   quick test, remember to set the environment variable `CIFAR100` to
   the directory where you extracted the cifar100 dataset, before you
   run `train_val.py`.
   
2. `imagenet`: Imagenet training and validation with a vgg-16
   architecture network. Start by looking at `train.py` /
   `train_avg.py` and `val.py`. You'll need to create the `train.txt`
   and `val.txt` with lists of JPEG file names and labels: you can use
   `data/mk_data.sh` to do this.
