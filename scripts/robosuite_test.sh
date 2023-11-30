#!/bin/bash
cudagpus=${2:-$(nvidia-smi --list-gpus | wc -l)}
# gpulist=($(seq 0 1 $cudagpus))
gpulist=(3)

numgpus=${#gpulist[@]}
# robosuite 1m settings
tasklist=(Door)

expname="robosuite_test"
totaltimesteps="1000"
buffersize="500"
learningstarts="300"

mkdir -p logs/${expname}
mkdir -p recordings/${expname}
mkdir -p trained_models/${expname}

for i in ${!tasklist[@]}
do
    gpuindex=$(( $i % $numgpus ))
    gpuid=${gpulist[$gpuindex]}
    (
        for seed in 0
        do
            echo "${expname} GPU: ${gpuid} Env: ${tasklist[$i]} Seed: ${seed} ${1}"
            # sleep 5
            basename=$(basename $1)
            echo "========" # >> logs/${expname}/${envlist[$i]}__${basename}__${seed}.txt
            CUDA_VISIBLE_DEVICES=$gpuid python $1 --task-name ${tasklist[$i]} \
                                                  --seed $seed --exp-name $expname \
                                                  --capture-video \
                                                  --total-timesteps $totaltimesteps --buffer-size $buffersize \
                                                  --learning-starts $learningstarts \
                                                  --eval-num 1 \
                                                  ${@:3} # >> logs/${expname}/${envlist[$i]}__${basename}__${seed}.txt
        done
    ) &
done
wait