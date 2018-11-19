#!/usr/bin/env bash
# usage: ./infer-alchemy2.sh project meme1 [meme2 ...]
project=$1
echo "=============== running Alchemy2 MLN inference for memes of project " $project "..."

while true
do

    meme=$2
    if [ "$meme" = "" ]
    then
        break
    fi

    echo "=============== running inference on meme " $meme "..."
    ~/social/alchemy-2/bin/infer -i data/$project/weights-$project-alchemy2.mln \
                                 -e data/$project/evidence-alchemy2/ev-test-$project-alchemy2-m$meme.db \
                                 -r data/$project/results-alchemy2/results-$project-alchemy2-m$meme.results \
                                 -q isActivated \
                                 -maxSteps 1000
                                 2>&1 | tee log/infer-alchemy2-m$meme.log
    shift

done