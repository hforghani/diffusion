#!/bin/bash

project_num=100
db="weibo"
min_depth=2
min=10
max=500

for i in $(seq 1 $project_num); do
  range=$(($max - $min))
  num=$(($min + $RANDOM % $range))
  total_num=${num%.*}
  project="$db-analysis-$i"
  echo "running : python3 sampledata.py -d $db -D $min_depth -n $total_num -p $project"
  python3 sampledata.py -d $db -D $min_depth -n $total_num -p $project
  python3 project_stat.py $project
done
