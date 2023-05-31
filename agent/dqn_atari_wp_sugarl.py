import argparse
import os, sys
import os.path as osp
import random
import time
from itertools import product
from distutils.util import strtobool

sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize

from common.buffer import DoubleActionReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari
from torch.utils.tensorboard import SummaryWriter

from active_gym import AtariFixedFovealPeripheralEnv, AtariEnvArgs


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
    parser.add_argument("--sensory-action-mode", type=str, default="absolute")
    parser.add_argument("--sensory-action-space", type=int, default=10) # ignored when sensory_action_mode="relative"
    parser.add_argument("--resize-to-full", default=False, action="store_true")
    parser.add_argument("--peripheral-res", type=int, default=20)
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
        env = AtariFixedFovealPeripheralEnv(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor_action"].n
            if sensory_action_set is not None:
                sensory_action_space_size = len(sensory_action_set)
            else:
                sensory_action_space_size = env.single_action_space["sensory_action"].n
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.motor_action_head = nn.Linear(512, motor_action_space_size)
        self.sensory_action_head = None
        if sensory_action_space_size is not None:
            self.sensory_action_head = nn.Linear(512, sensory_action_space_size)
        

    def forward(self, x):
        x = self.backbone(x)
        motor_action = self.motor_action_head(x)
        sensory_action = None
        if self.sensory_action_head:
            sensory_action = self.sensory_action_head(x)
        return motor_action, sensory_action

class SelfPredictionNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor_action"].n
            if sensory_action_set is not None:
                sensory_action_space_size = len(sensory_action_set)
            else:
                sensory_action_space_size = env.single_action_space["sensory_action"].n
        
        self.backbone = nn.Sequential(
            nn.Conv2d(8, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512, motor_action_space_size),
        )

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, x, target) -> torch.Tensor:
        return self.loss(x, target)
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


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

    # env setup
    envs = []
    for i in range(args.env_num):
        envs.append(make_env(args.env, args.seed+i, frame_stack=args.frame_stack, action_repeat=args.action_repeat,
                                fov_size=(args.fov_size, args.fov_size), 
                                fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                                peripheral_res=(args.peripheral_res, args.peripheral_res),
                                sensory_action_mode=args.sensory_action_mode,
                                sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
                                resize_to_full=args.resize_to_full,
                                clip_reward=args.clip_reward,
                                mask_out=True))
    # envs = gym.vector.AsyncVectorEnv(envs)
    envs = gym.vector.SyncVectorEnv(envs)

    sugarl_r_scale = get_sugarl_reward_scale_atari(args.env)

    resize = Resize((84, 84))

    # get a discrete observ action space
    OBSERVATION_SIZE = (84, 84)
    observ_x_max, observ_y_max = OBSERVATION_SIZE[0]-args.fov_size, OBSERVATION_SIZE[1]-args.fov_size
    sensory_action_step = (observ_x_max//args.sensory_action_x_size,
                          observ_y_max//args.sensory_action_y_size)
    sensory_action_x_set = list(range(0, observ_x_max, sensory_action_step[0]))[:args.sensory_action_x_size]
    sensory_action_y_set = list(range(0, observ_y_max, sensory_action_step[1]))[:args.sensory_action_y_size]
    sensory_action_set = [np.array(a) for a in list(product(sensory_action_x_set, sensory_action_y_set))]

    q_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    target_network.load_state_dict(q_network.state_dict())

    sfn = SelfPredictionNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    sfn_optimizer = optim.Adam(sfn.parameters(), lr=args.learning_rate)

    rb = DoubleActionReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space["motor_action"],
        Discrete(len(sensory_action_set)),
        device,
        n_envs=envs.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset()
    global_transitions = 0
    pvm_buffer = PVMBuffer(args.pvm_stack, (envs.num_envs, args.frame_stack,)+OBSERVATION_SIZE)

    while global_transitions < args.total_timesteps:
        pvm_buffer.append(obs)
        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_transitions)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            motor_actions = np.array([actions[0]["motor_action"]])
            sensory_actions = np.array([random.randint(0, len(sensory_action_set)-1)])
        else:
            motor_q_values, sensory_q_values = q_network(resize(torch.from_numpy(pvm_obs)).to(device))
            motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
            sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step({"motor_action": motor_actions, 
                        "sensory_action": [sensory_action_set[a] for a in  sensory_actions] })
        # print (global_step, infos)

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
        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(real_next_obs)
        real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
        rb.add(pvm_obs, real_next_pvm_obs, motor_actions, sensory_actions, rewards, dones, {})

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # INC total transitions 
        global_transitions += args.env_num

        obs_backup = obs # back obs

        if global_transitions < args.batch_size:
            continue

        # Training
        if global_transitions % args.train_frequency == 0:
            data = rb.sample(args.batch_size // args.env_num) # counter-balance the true global transitions used for training

            # sfn learning
            concat_observation = torch.concat([data.next_observations, data.observations], dim=1) # concat at dimension T
            pred_motor_actions = sfn(resize(concat_observation))
            # print (pred_motor_actions.size(), data.motor_actions.size())
            sfn_loss = sfn.get_loss(pred_motor_actions, data.motor_actions.flatten())
            sfn_optimizer.zero_grad()
            sfn_loss.backward()
            sfn_optimizer.step()
            observ_r = F.softmax(pred_motor_actions).gather(1, data.motor_actions).squeeze().detach() # 0-1

            # Q network learning
            if global_transitions > args.learning_starts:
                with torch.no_grad():
                    motor_target, sensory_target = target_network(resize(data.next_observations))
                    motor_target_max, _ = motor_target.max(dim=1)
                    sensory_target_max, _ = sensory_target.max(dim=1)
                    # scale step-wise reward with observ_r
                    observ_r_adjusted = observ_r.clone()
                    observ_r_adjusted[data.rewards.flatten() > 0] = 1 - observ_r_adjusted[data.rewards.flatten() > 0]
                    td_target = data.rewards.flatten() - (1 - observ_r) * sugarl_r_scale + args.gamma * (motor_target_max+sensory_target_max) * (1 - data.dones.flatten())
                    original_td_target = data.rewards.flatten() + args.gamma * (motor_target_max+sensory_target_max) * (1 - data.dones.flatten())

                old_motor_q_value, old_sensory_q_val = q_network(resize(data.observations))
                old_motor_val = old_motor_q_value.gather(1, data.motor_actions).squeeze()
                old_sensory_val = old_sensory_q_val.gather(1, data.sensory_actions).squeeze()
                old_val = old_motor_val + old_sensory_val
                
                loss = F.mse_loss(td_target, old_val)

                if global_transitions % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_transitions)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_transitions)
                    writer.add_scalar("losses/motor_q_values", old_motor_val.mean().item(), global_transitions)
                    writer.add_scalar("losses/action_q_values", old_sensory_val.mean().item(), global_transitions)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_transitions / (time.time() - start_time)), global_transitions)

                    writer.add_scalar("losses/sfn_loss", sfn_loss.item(), global_transitions)
                    writer.add_scalar("losses/observ_r", observ_r.mean().item(), global_transitions)
                    writer.add_scalar("losses/original_td_target", original_td_target.mean().item(), global_transitions)
                    writer.add_scalar("losses/sugarl_r_scaled_td_target", td_target.mean().item(), global_transitions)

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
                sfn.eval()
                
                eval_episodic_returns, eval_episodic_lengths = [], []

                for eval_ep in range(args.eval_num):
                    eval_env = [make_env(args.env, args.seed+eval_ep, frame_stack=args.frame_stack, action_repeat=args.action_repeat, 
                            fov_size=(args.fov_size, args.fov_size), 
                            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                            peripheral_res=(args.peripheral_res, args.peripheral_res),
                            sensory_action_mode=args.sensory_action_mode,
                            sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
                            resize_to_full=args.resize_to_full,
                            clip_reward=args.clip_reward,
                            mask_out=True,
                            training=False,
                            record=args.capture_video)]
                    eval_env = gym.vector.SyncVectorEnv(eval_env)
                    obs_eval, _ = eval_env.reset()
                    done = False
                    pvm_buffer_eval = PVMBuffer(args.pvm_stack, (eval_env.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
                    while not done:
                        pvm_buffer_eval.append(obs_eval)
                        pvm_obs_eval = pvm_buffer_eval.get_obs(mode="stack_max")
                        motor_q_values, sensory_q_values = q_network(resize(torch.from_numpy(pvm_obs_eval)).to(device))
                        motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
                        sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()
                        next_obs_eval, rewards, dones, _, infos = eval_env.step({"motor_action": motor_actions, 
                                                                            "sensory_action": [sensory_action_set[a] for a in sensory_actions]})
                        obs_eval = next_obs_eval
                        done = dones[0]
                        if done:
                            eval_episodic_returns.append(infos['final_info'][0]["reward"])
                            eval_episodic_lengths.append(infos['final_info'][0]["ep_len"])
                            if args.capture_video:
                                record_file_dir = os.path.join("recordings", args.exp_name, os.path.basename(__file__).rstrip(".py"), args.env)
                                os.makedirs(record_file_dir, exist_ok=True)
                                record_file_fn = f"{args.env}_seed{args.seed}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                eval_env.envs[0].save_record_to_file(os.path.join(record_file_dir, record_file_fn))
                                if eval_ep == 0:
                                    model_file_dir = os.path.join("trained_models", args.exp_name, os.path.basename(__file__).rstrip(".py"), args.env)
                                    os.makedirs(model_file_dir, exist_ok=True)
                                    model_fn = f"{args.env}_seed{args.seed}_step{global_transitions:07d}_model.pt"
                                    torch.save({"sfn": sfn.state_dict(), "q": q_network.state_dict()}, os.path.join(model_file_dir, model_fn))
                                

                writer.add_scalar("charts/eval_episodic_return", np.mean(eval_episodic_returns), global_transitions)
                writer.add_scalar("charts/eval_episodic_return_std", np.std(eval_episodic_returns), global_transitions)
                # writer.add_scalar("charts/eval_episodic_length", np.mean(), global_transitions)
                print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}] [R list: {','.join([str(r) for r in eval_episodic_returns])}]")

                q_network.train()
                sfn.train()
        
        obs = obs_backup # restore obs if eval occurs



    envs.close()
    eval_env.close()
    writer.close()