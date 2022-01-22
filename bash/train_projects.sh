#!/bin/bash

project_num=100
db="weibo"
methods="parentmemm redtdmemm aslt"

for i in $(seq 14 $project_num); do
  for method in $methods; do
      project="$db-analysis-$i"
      command="predict.py -p $project -m $method -i 1 -d 2 -t 0.5 --multiprocessed"
      echo "running : $command"
      $command
    done
done
