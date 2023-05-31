#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

# dmc 1m settings
domainlist=(reacher) # reacher walker humanoid cheetah dog)
tasklist=(easy) # hard walk walk run fetch)

expname="dmc_test"
totaltimesteps="1000"
buffersize="500"
learningstarts="300"

mkdir -p logs/${expname}
mkdir -p recordings/${expname}
mkdir -p trained_models/${expname}

for i in ${!domainlist[@]}
do
    gpuid=$(( $i % $numgpus ))
    (
        for seed in 0
        do
            echo "${expname} GPU: ${gpuid} Env: ${envlist[$i]} Seed: ${seed} ${1}"
            # sleep 5
            basename=$(basename $1)
            echo "========" #>> logs/${expname}/${envlist[$i]}__${basename}__${seed}.txt
            CUDA_VISIBLE_DEVICES=$gpuid python $1 --domain-name ${domainlist[$i]} --task-name ${tasklist[$i]} \
                                                  --seed $seed --exp-name $expname \
                                                  --fov-size 20 \
                                                  --capture-video \
                                                  --total-timesteps $totaltimesteps --buffer-size $buffersize \
                                                  --learning-starts $learningstarts \
                                                  ${@:3} #>> logs/${expname}/${envlist[$i]}__${basename}__${seed}.txt
        done
    ) &
done
wait