#!/usr/bin/env bash
# usage: ./infer-alchemy2.sh project cascade1 [cascade2 ...]
project=$1
echo "=============== running Alchemy2 MLN inference for cascades of project " $project "..."

while true
do

    cascade=$2
    if [ "$cascade" = "" ]
    then
        break
    fi

    echo "=============== running inference on cascade " $cascade "..."
    ~/social/alchemy-2/bin/infer -i ../data/$project/weights-$project-alchemy2.mln \
                                 -e ../data/$project/evidence-alchemy2/ev-test-$project-alchemy2-$cascade.db \
                                 -r ../data/$project/results-alchemy2/results-$project-alchemy2-$cascade.results \
                                 -q isActivated \
                                 -maxSteps 1000
                                 2>&1 | tee log/infer-alchemy2-$cascade.log
    shift

done