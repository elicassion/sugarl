import argparse
import os, sys
import os.path as osp
import random
import time
from collections import deque
from itertools import product
from distutils.util import strtobool

sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize

from common.buffer import ReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything
from torch.utils.tensorboard import SummaryWriter

from active_gym import AtariFixedFovealEnv, AtariEnvArgs



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # env setting
    parser.add_argument("--env", type=str, default="breakout",
        help="the id of the environment")
    parser.add_argument("--env-num", type=int, default=1, 
        help="# envs in parallel")
    parser.add_argument("--frame-stack", type=int, default=4,
        help="frame stack #")
    parser.add_argument("--action-repeat", type=int, default=4,
        help="action repeat #")
    parser.add_argument("--clip-reward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # fov setting
    parser.add_argument("--fov-size", type=int, default=50)
    parser.add_argument("--fov-init-loc", type=int, default=0)
    parser.add_argument("--sensory-action-mode", type=str, default="relative")
    parser.add_argument("--sensory-action-space", type=int, default=10) # ignored when sensory_action_mode="relative"
    parser.add_argument("--resize-to-full", default=False, action="store_true")
    # for discrete observ action
    parser.add_argument("--sensory-action-x-size", type=int, default=4)
    parser.add_argument("--sensory-action-y-size", type=int, default=4)
    # pvm setting
    parser.add_argument("--pvm-stack", type=int, default=3)

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=3000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=500000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")

    # eval args
    parser.add_argument("--eval-frequency", type=int, default=-1,
        help="eval frequency. default -1 is eval at the end.")
    parser.add_argument("--eval-num", type=int, default=10,
        help="eval frequency. default -1 is eval at the end.")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_name, seed, **kwargs):
    def thunk():
        env_args = AtariEnvArgs(
            game=env_name, seed=seed, obs_size=(84, 84), **kwargs
        )
        env = AtariFixedFovealEnv(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, override_action_set=None):
        super().__init__()
        if override_action_set:
            action_space_size = override_action_set.n
        else:
            if isinstance(env.single_action_space, Discrete):
                action_space_size = env.single_action_space.n
            elif isinstance(env.single_action_space, Dict):
                action_space_size = env.single_action_space["motor_action"].n
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_size),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)





if __name__ == "__main__":
    args = parse_args()
    args.env = args.env.lower()
    run_name = f"{args.env}__{os.path.basename(__file__)}__{args.seed}__{get_timestr()}"
    run_dir = os.path.join("runs", args.exp_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(run_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    seed_everything(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # get a discrete observ action space
    OBSERVATION_SIZE = (84, 84)
    observ_x_max, observ_y_max = OBSERVATION_SIZE[0]-args.fov_size, OBSERVATION_SIZE[1]-args.fov_size
    sensory_action_step = (observ_x_max//args.sensory_action_x_size,
                          observ_y_max//args.sensory_action_y_size)
    sensory_action_set = [(-sensory_action_step[0], 0),
                         (sensory_action_step[0], 0),
                         (0, 0),
                         (0, -sensory_action_step[1]),
                         (0, sensory_action_step[1])]

    # env setup
    envs = []
    for i in range(args.env_num):
        envs.append(make_env(args.env, args.seed+i, frame_stack=args.frame_stack, action_repeat=args.action_repeat,
                                fov_size=(args.fov_size, args.fov_size), 
                                fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                                sensory_action_mode=args.sensory_action_mode,
                                sensory_action_space=(-max(sensory_action_step), max(sensory_action_step)),
                                resize_to_full=args.resize_to_full,
                                clip_reward=args.clip_reward,
                                mask_out=True))
    # envs = gym.vector.AsyncVectorEnv(envs)
    envs = gym.vector.SyncVectorEnv(envs)

    resize = Resize((84, 84))

    # make motor sensory joint action space
    motor_action_set = list(range(envs.single_action_space["motor_action"].n))
    motor_sensory_joint_action_set = []
    for ma in motor_action_set:
        for sa in sensory_action_set:
            motor_sensory_joint_action_set.append((ma, *sa))
    motor_sensory_joint_action_space = Discrete(len(motor_sensory_joint_action_set), seed=args.seed)

    # make a method to seperate joint action
    def seperate_motor_sensory_joint_action(msas: np.ndarray):
        mas, sas = [], []
        for msa in msas:
            msa = motor_sensory_joint_action_set[msa]
            mas.append(msa[0])
            sas.append(msa[1:])
        mas = np.array(mas)
        sas = np.array(sas)
        return mas, sas
        


    q_network = QNetwork(envs, motor_sensory_joint_action_space).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, motor_sensory_joint_action_space).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        motor_sensory_joint_action_space,
        device,
        n_envs=envs.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    global_transitions = 0
    pvm_buffer = PVMBuffer(args.pvm_stack, (envs.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
    
    while global_transitions < args.total_timesteps:
        pvm_buffer.append(obs)
        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_transitions)
        if random.random() < epsilon:
            actions = np.array([motor_sensory_joint_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(resize(torch.from_numpy(pvm_obs)).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        # print (actions)
        motor_actions, sensory_actions = seperate_motor_sensory_joint_action(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step({"motor_action": motor_actions, 
                        "sensory_action": sensory_actions })
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for idx, d in enumerate(dones):
                if d:
                    print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [R: {infos['final_info'][idx]['reward']:.2f}]")
                    writer.add_scalar("charts/episodic_return", infos['final_info'][idx]["reward"], global_transitions)
                    writer.add_scalar("charts/episodic_length", infos['final_info'][idx]["ep_len"], global_transitions)
                    writer.add_scalar("charts/epsilon", epsilon, global_transitions)
                    break
        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
                fov_idx = random.randint(0, len(sensory_action_set)-1)
        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(real_next_obs)
        real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
        rb.add(pvm_obs, real_next_pvm_obs, actions, rewards, dones, {})

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        
        # INC total transitions 
        global_transitions += args.env_num


        obs_backup = obs # back obs
        # ALGO LOGIC: training.
        if global_transitions > args.learning_starts:
            if global_transitions % args.train_frequency == 0:
                data = rb.sample(args.batch_size // args.env_num) # counter-balance the true global transitions used for training
                with torch.no_grad():
                    target_max, _ = target_network(resize(data.next_observations)).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(resize(data.observations)).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_transitions % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_transitions)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_transitions)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_transitions / (time.time() - start_time)), global_transitions)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if (global_transitions // args.env_num) % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            # evaluation
            if (global_transitions % args.eval_frequency == 0 and args.eval_frequency > 0) or \
               (global_transitions >= args.total_timesteps):
                q_network.eval()
                
                eval_episodic_returns, eval_episodic_lengths = [], []

                for eval_ep in range(args.eval_num):
                    eval_env = [make_env(args.env, args.seed+eval_ep, frame_stack=args.frame_stack, action_repeat=args.action_repeat, 
                            fov_size=(args.fov_size, args.fov_size), 
                            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                            sensory_action_mode=args.sensory_action_mode,
                            sensory_action_space=(-max(sensory_action_step), max(sensory_action_step)),
                            resize_to_full=args.resize_to_full,
                            clip_reward=args.clip_reward,
                            training=False,
                            mask_out=True)]
                    eval_env = gym.vector.SyncVectorEnv(eval_env)
                    obs, _ = eval_env.reset()
                    done = False
                    pvm_buffer = PVMBuffer(args.pvm_stack, (eval_env.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
                    while not done:
                        pvm_buffer.append(obs)
                        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
                        q_values = q_network(resize(torch.from_numpy(pvm_obs)).to(device))
                        actions = torch.argmax(q_values, dim=1).cpu().numpy()
                        motor_actions, sensory_actions = seperate_motor_sensory_joint_action(actions)
                        next_obs, rewards, dones, _, infos = eval_env.step({"motor_action": motor_actions, 
                                                                            "sensory_action": sensory_actions})
                        obs = next_obs
                        done = dones[0]
                        if done:
                            eval_episodic_returns.append(infos['final_info'][0]["reward"])
                            eval_episodic_lengths.append(infos['final_info'][0]["ep_len"])

                            if args.capture_video:
                                record_file_dir = os.path.join("recordings", args.exp_name, os.path.basename(__file__).rstrip(".py"))
                                os.makedirs(record_file_dir, exist_ok=True)
                                record_file_fn = f"seed{args.seed}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                eval_env.envs[0].save_record_to_file(os.path.join(record_file_dir, record_file_fn))
                                if eval_ep == 0:
                                    model_file_dir = os.path.join("trained_models", args.exp_name, os.path.basename(__file__).rstrip(".py"), args.env)
                                    os.makedirs(model_file_dir, exist_ok=True)
                                    model_fn = f"{args.env}_seed{args.seed}_step{global_transitions:07d}_model.pt"
                                    torch.save({"sfn": None, "q": q_network.state_dict()}, os.path.join(model_file_dir, model_fn))

                writer.add_scalar("charts/eval_episodic_return", np.mean(eval_episodic_returns), global_transitions)
                writer.add_scalar("charts/eval_episodic_return_std", np.std(eval_episodic_returns), global_transitions)
                print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}] [R list: {','.join([str(r) for r in eval_episodic_returns])}]")

                q_network.train()
        obs = obs_backup # restore obs if eval occurs


    envs.close()
    eval_env.close()
    writer.close()
