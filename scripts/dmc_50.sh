#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

# dmc 1m settings
domainlist=(ball_in_cup cartpole cheetah dog fish walker)
tasklist=(catch swingup run fetch swim walk)

expname="dmc_50"
totaltimesteps="100000"
buffersize="100000"
learningstarts="2000"

mkdir -p logs/${expname}
mkdir -p recordings/${expname}
mkdir -p trained_models/${expname}

jobnum=0

for i in ${!domainlist[@]}
do
    for seed in 0 1 2 3 4
    do
        gpuid=$(( $jobnum % $numgpus ))
        echo "${expname} GPU: ${gpuid} Env: ${domainlist[$i]}-${tasklist[$i]} Seed: ${seed} ${1}"
        # sleep 5
        basename=$(basename $1)
        echo "========" >> logs/${expname}/${domainlist[$i]}-${tasklist[$i]}__${basename}__${seed}.txt
        MUJOCO_EGL_DEVICE_ID=$gpuid CUDA_VISIBLE_DEVICES=$gpuid python $1 --domain-name ${domainlist[$i]} --task-name ${tasklist[$i]} \
                                                --seed $seed --exp-name $expname \
                                                --fov-size 50 \
                                                --capture-video \
                                                --total-timesteps $totaltimesteps --buffer-size $buffersize \
                                                --learning-starts $learningstarts \
                                                --eval-frequency 10000 \
                                                ${@:3} >> logs/${expname}/${domainlist[$i]}-${tasklist[$i]}__${basename}__${seed}.txt &
        ((jobnum=jobnum+1))
    done
done
wait