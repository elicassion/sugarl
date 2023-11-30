#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}
# numgpus=6
gpustart=0

# robosuite 1m settings
tasklist=(Wipe Stack NutAssemblySquare Door Lift)

expname="robosuite"
totaltimesteps="100000"
buffersize="100000"
learningstarts="2000"

mkdir -p logs/${expname}
mkdir -p recordings/${expname}
mkdir -p trained_models/${expname}

jobnum=0+$gpustart

for i in ${!tasklist[@]}
do
    for seed in 0 1 2
    do
        gpuid=$(( $jobnum % $numgpus ))
        echo "${expname} GPU: ${gpuid} Env: ${tasklist[$i]} Seed: ${seed} ${1}"
        # sleep 5
        basename=$(basename $1)
        echo "========" >> logs/${expname}/${tasklist[$i]}__${basename}__${seed}.txt
        CUDA_VISIBLE_DEVICES=$gpuid python $1 --task-name ${tasklist[$i]} \
                                                --seed $seed --exp-name $expname \
                                                --capture-video \
                                                --total-timesteps $totaltimesteps --buffer-size $buffersize \
                                                --learning-starts $learningstarts \
                                                --eval-num 10 \
                                                ${@:3} >> logs/${expname}/${tasklist[$i]}__${basename}__${seed}.txt &
        ((jobnum=jobnum+1))
    done
done
wait
