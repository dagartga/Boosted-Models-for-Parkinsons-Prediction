#!/bin/sh

python train.py --fold 0 --model rf_reg --target updrs_1
python train.py --fold 1 --model rf_reg --target updrs_1
python train.py --fold 2 --model rf_reg --target updrs_1
python train.py --fold 3 --model rf_reg --target updrs_1
python train.py --fold 4 --model rf_reg --target updrs_1