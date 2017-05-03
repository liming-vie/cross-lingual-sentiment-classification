#!/bin/bash

if [ $# != 3 ]; then
    echo 'Usage sh train.sh p n num_iter'
    exit 1
fi

p=$1
n=$2
num_iter=$3

python co_training.py dvd $p $n $num_iter 
python co_training.py music $p $n $num_iter
python co_training.py book $p $n $num_iter
