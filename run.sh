#!/bin/bash

ulimit -n 90000
python main.py --dataset cifar10 --dataroot data --outf output --cuda