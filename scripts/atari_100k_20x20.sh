#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

# atari 100k settings
envlist=(alien amidar assault asterix bank_heist battle_zone boxing breakout chopper_command crazy_climber demon_attack freeway frostbite gopher hero jamesbond kangaroo krull kung_fu_master ms_pacman pong private_eye qbert road_runner seaquest up_n_down) #pong qbert seaquest zaxxon

expname="atari_100k_20x20"
totaltimesteps="1000000"
buffersize="100000"
learningstarts="80000"

mkdir -p logs/${expname}
mkdir -p recordings/${expname}
mkdir -p trained_models/${expname}

for i in ${!envlist[@]}
do
    gpuid=$(( $i % $numgpus ))
    (
        for seed in 0 1 2 3 4
        do
            echo "${expname} GPU: ${gpuid} Env: ${envlist[$i]} Seed: ${seed} ${1}"
            # sleep 5
            basename=$(basename $1)
            echo "========" >> logs/${expname}/${envlist[$i]}__${basename}__${seed}.txt
            CUDA_VISIBLE_DEVICES=$gpuid python $1 --env ${envlist[$i]} --seed $seed --exp-name ${expname} \
                                                  --fov-size 20 \
                                                  --clip-reward \
                                                  --capture-video \
                                                  --total-timesteps $totaltimesteps --buffer-size $buffersize \
                                                  --learning-starts $learningstarts \
                                                  ${@:2} >> logs/${expname}/${envlist[$i]}__${basename}__${seed}.txt
        done
    ) &
done
wait