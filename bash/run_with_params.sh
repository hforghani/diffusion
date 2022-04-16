#!/bin/bash

project="twitter-size77"

linesearch="MoreThuente"
for c1 in 0 .02 .04 .06 .08 .1; do
  for c2 in .2 .22 .24 .26 .28 .3; do
      python3 predict.py -p $project -m crf -i 0 --validation --param threshold 0 1 0.01 \
        --param algorithm lbfgs \
        --param c1 $c1 \
        --param c2 $c2 \
        --param linesearch $linesearch \
        --multiprocessed
  done
done

#for c2 in 1.3 1.4 1.5 1.6 1.7; do
#  for calibration_rate in 1.5 2 2.5 3; do
#    python3 predict.py -p $project -m crf -i 0 --validation --param threshold 0 1 0.01 \
#      --param algorithm l2sgd \
#      --param c2 $c2 \
#      --param calibration_rate $calibration_rate \
#      --multiprocessed
#  done
#done

#error_sensitive=1
#averaging=1
#for pa_type in 0 1 2; do
#  for c in .5 .6 .7 .8 .9 1 1.1 1.2 1.3 1.4; do
#        python3 predict.py -p $project -m crf -i 0 --validation --param threshold 0 1 0.01 \
#          --param algorithm pa \
#          --param pa_type $pa_type \
#          --param c $c \
#          --param error_sensitive $error_sensitive \
#          --param averaging $averaging \
#          --multiprocessed
#  done
#done

#for variance in 0.5 1 1.5 2; do
#  for gamma in 0 0.5 1; do
#    python3 predict.py -p $project -m crf -i 0 --validation --param threshold 0 1 0.01 \
#      --param algorithm arow \
#      --param variance $variance \
#      --param gamma $gamma \
#      --multiprocessed
#  done
#done
