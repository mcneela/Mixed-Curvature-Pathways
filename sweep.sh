#!/bin/bash

DIRS="input_data/*"
for dir in $DIRS do
    f="input_data/${dir}/${dir}.txt"
    python utils.py --path $f
    o="input_data/${dir}/${dir}.edges"
    python pytorch/pytorch_hyperbolic.py --dataset $o --batch-size 32 --epochs 1000  --checkpoint-freq 100 --subsample 16 -l .001 --euc 3 --edim 33 --hyp 3 --dim 33 --sdim 33 --sph 3
    