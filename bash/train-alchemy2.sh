#!/usr/bin/env bash
# usage: ./infer-alchemy2.sh project
project=$1
log_file="log/train-alchemy2-$project.log"

#echo "=============== creating rules and evidences for project " $project "..."
#
#python3 ../createrules.py -p $project -f alchemy2 2>&1 | tee $log_file
#python3 ../createevidence.py -p $project -f alchemy2 -s train 2>&1 | tee $log_file
#python3 ../createevidence.py -p $project -f alchemy2 -s test --multiple 2>&1 | tee $log_file

echo "=============== training Alchemy2 MLN for project " $project "..."
~/social/code/alchemy-2/bin/learnwts -g \
                                -i ../data/$project/tolearn-$project-alchemy2.mln \
                                -o ../data/$project/weights-$project-alchemy2.mln \
                                -t ../data/$project/evidence-alchemy2/ev-train-$project-alchemy2.db \
                                2>&1 | tee $log_file
