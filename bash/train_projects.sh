#!/bin/bash

project_num=100
db="weibo"
methods="tdmemm aslt"

for i in $(seq 1 $project_num); do
  for method in $methods; do
      project="$db-analysis-$i"
      if [ "$method" == "tdmemm" ]; then
        iterations=10
      else
        iterations=2
      fi
      command="python3 predict.py -p $project -m $method -i 1 -d 2 -t 0.5 --multiprocessed --iterations $iterations"
      echo "running : $command"
      $command
    done
done
