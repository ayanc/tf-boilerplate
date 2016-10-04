#!/bin/bash

if [[ $# != 1 ]]; then
    echo USAGE: $0 imagenet_dir
    echo Call with directory whose subdirs train and val contain
    echo the training and validation sets.
    exit -1
fi

lbl=0
rm train.txt val.txt 2>/dev/null
for c in `cat class_dirs.txt`; do
    echo Processing class $lbl
    ls $1/val/$c/* | while read line; do
	echo $line $lbl >> val.txt
    done
    ls $1/train/$c/* | while read line; do
	echo $line $lbl >> train.txt
    done
    lbl=$((lbl+1))
done
    

