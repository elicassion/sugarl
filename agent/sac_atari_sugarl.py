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
from torch.distributions.categorical import Categorical

from common.buffer import DoubleActionReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari
from torch.utils.tensorboard import SummaryWriter

from active_gym import AtariFixedFovealEnv, AtariEnvArgs


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__),
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
    # for discrete observ action
    parser.add_argument("--sensory-action-x-size", type=int, default=4)
    parser.add_argument("--sensory-action-y-size", type=int, default=4)
    # pvm setting
    parser.add_argument("--pvm-stack", type=int, default=3)

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.02,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--sensory-alpha-autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--motor-target-entropy-scale", type=float, default=0.2,
        help="coefficient for scaling the autotune entropy target")
    parser.add_argument("--sensory-target-entropy-scale", type=float, default=0.2,
        help="coefficient for scaling the autotune entropy target")

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


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
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


class Actor(nn.Module):
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
        motor_action_logits = self.motor_action_head(x)
        sensory_action_logits = None
        if self.sensory_action_head:
            sensory_action_logits = self.sensory_action_head(x)

        return motor_action_logits, sensory_action_logits

    def get_action_from_logits(self, logits) -> dict:
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return {"action": action, 
                "log_prob": log_prob, 
                "action_probs": action_probs}

    def get_action(self, x):
        motor_action_logits, sensory_action_logits = self(x)
        motor_action = self.get_action_from_logits(motor_action_logits)
        sensory_action = self.get_action_from_logits(sensory_action_logits)
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

    actor = Actor(envs, sensory_action_set=sensory_action_set).to(device)
    qf1 = SoftQNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    qf2 = SoftQNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    qf1_target = SoftQNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    qf2_target = SoftQNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        motor_target_entropy = -args.motor_target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space["motor_action"].n))
        motor_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        motor_alpha = motor_log_alpha.exp().item()
        motor_a_optimizer = optim.Adam([motor_log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        motor_alpha = args.alpha


    if args.sensory_alpha_autotune:
        sensory_target_entropy = -args.sensory_target_entropy_scale * torch.log(1 / torch.tensor(len(sensory_action_set)))
        sensory_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        sensory_alpha = sensory_log_alpha.exp().item()
        sensory_a_optimizer = optim.Adam([sensory_log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        sensory_alpha = args.alpha

    sfn = SelfPredictionNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    sfn_optimizer = optim.Adam(sfn.parameters(), lr=args.policy_lr)

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
        if global_transitions < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            motor_actions = np.array([actions[0]["motor_action"]])
            sensory_actions = np.array([random.randint(0, len(sensory_action_set)-1)])
        else:
            motor_actions_pack, sensory_actions_pack = actor.get_action(resize(torch.from_numpy(pvm_obs)).to(device))
            motor_actions = motor_actions_pack["action"].detach().cpu().numpy()
            sensory_actions = sensory_actions_pack["action"].detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step({"motor_action": motor_actions, 
                        "sensory_action": [sensory_action_set[a] for a in  sensory_actions] })

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for idx, d in enumerate(dones):
                if d:
                    print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [R: {infos['final_info'][idx]['reward']:.2f}]")
                    writer.add_scalar("charts/episodic_return", infos['final_info'][idx]["reward"], global_transitions)
                    writer.add_scalar("charts/episodic_length", infos['final_info'][idx]["ep_len"], global_transitions)
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

        # ALGO LOGIC: training.
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

            if global_transitions > args.learning_starts:
                # CRITIC training
                with torch.no_grad():
                    motor_actions_pack, sensory_actions_pack = actor.get_action(data.next_observations)
                    motor_next_state_log_pi = motor_actions_pack["log_prob"]
                    motor_next_state_action_probs = motor_actions_pack["action_probs"]
                    sensory_next_state_log_pi = sensory_actions_pack["log_prob"]
                    sensory_next_state_action_probs = sensory_actions_pack["action_probs"]
                    # _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    motor_qf1_next_target, sensory_qf1_next_target = qf1_target(data.next_observations)
                    motor_qf2_next_target, sensory_qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    motor_min_qf_next_target = motor_next_state_action_probs * (
                        torch.min(motor_qf1_next_target, motor_qf2_next_target) - motor_alpha * motor_next_state_log_pi
                    )
                    sensory_min_qf_next_target = sensory_next_state_action_probs * (
                        torch.min(sensory_qf1_next_target, sensory_qf2_next_target) - sensory_alpha * sensory_next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    motor_min_qf_next_target = motor_min_qf_next_target.sum(dim=1)
                    sensory_min_qf_next_target = sensory_min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() - (1 - observ_r) * sugarl_r_scale + (1 - data.dones.flatten()) * args.gamma * (motor_min_qf_next_target + sensory_min_qf_next_target)

                # use Q-values only for the taken actions
                motor_qf1_values, sensory_qf1_values = qf1(data.observations)
                motor_qf2_values, sensory_qf2_values = qf2(data.observations)
                motor_qf1_a_values = motor_qf1_values.gather(1, data.motor_actions.long()).view(-1)
                motor_qf2_a_values = motor_qf2_values.gather(1, data.motor_actions.long()).view(-1)
                sensory_qf1_a_values = sensory_qf1_values.gather(1, data.sensory_actions.long()).view(-1)
                sensory_qf2_a_values = sensory_qf2_values.gather(1, data.sensory_actions.long()).view(-1)
                qf1_loss = F.mse_loss(motor_qf1_a_values+sensory_qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(motor_qf2_a_values+sensory_qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                motor_actions_pack, sensory_actions_pack = actor.get_action(data.observations)
                motor_log_pi = motor_actions_pack["log_prob"]
                motor_action_probs = motor_actions_pack["action_probs"]
                sensory_log_pi = sensory_actions_pack["log_prob"]
                sensory_action_probs = sensory_actions_pack["action_probs"]
                with torch.no_grad():
                    motor_qf1_values, sensory_qf1_values = qf1(data.observations)
                    motor_qf2_values, sensory_qf2_values = qf2(data.observations)
                    motor_min_qf_values = torch.min(motor_qf1_values, motor_qf2_values)
                    sensory_min_qf_values = torch.min(sensory_qf1_values, sensory_qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                motor_actor_loss = (motor_action_probs * ((motor_alpha * motor_log_pi) - motor_min_qf_values)).mean()
                sensory_actor_loss = (sensory_action_probs * ((sensory_alpha * sensory_log_pi) - sensory_min_qf_values)).mean()
                actor_loss = motor_actor_loss + sensory_actor_loss

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    motor_alpha_loss = (motor_action_probs.detach() * (-motor_log_alpha * (motor_log_pi + motor_target_entropy).detach())).mean()
                    motor_a_optimizer.zero_grad()
                    motor_alpha_loss.backward()
                    motor_a_optimizer.step()
                    motor_alpha = motor_log_alpha.exp().item()

                if args.sensory_alpha_autotune:
                    sensory_alpha_loss = (sensory_action_probs.detach() * (-sensory_log_alpha * (sensory_log_pi + sensory_target_entropy).detach())).mean()
                    sensory_a_optimizer.zero_grad()
                    sensory_alpha_loss.backward()
                    sensory_a_optimizer.step()
                    sensory_alpha = sensory_log_alpha.exp().item()

                # update the target networks
                if global_transitions % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_transitions % 100 == 0:
                    # writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_transitions)
                    # writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_transitions)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_transitions)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_transitions)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_transitions)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_transitions)
                    writer.add_scalar("losses/motor_alpha", motor_alpha, global_transitions)
                    writer.add_scalar("losses/sensory_alpha", sensory_alpha, global_transitions)
                    # print("SPS:", int(global_transitions / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_transitions / (time.time() - start_time)), global_transitions)
                    if args.autotune:
                        writer.add_scalar("losses/motor_alpha_loss", motor_alpha_loss.item(), global_transitions)
                    if args.sensory_alpha_autotune:
                        writer.add_scalar("losses/sensory_alpha_loss", sensory_alpha_loss.item(), global_transitions)

            # evaluation
            if (global_transitions % args.eval_frequency == 0 and args.eval_frequency > 0) or \
               (global_transitions >= args.total_timesteps):
                qf1.eval()
                qf2.eval()
                actor.eval()
                sfn.eval()
                
                eval_episodic_returns, eval_episodic_lengths = [], []

                for eval_ep in range(args.eval_num):
                    eval_env = [make_env(args.env, args.seed+eval_ep, frame_stack=args.frame_stack, action_repeat=args.action_repeat, 
                            fov_size=(args.fov_size, args.fov_size), 
                            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                            sensory_action_mode=args.sensory_action_mode,
                            sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
                            resize_to_full=args.resize_to_full,
                            clip_reward=args.clip_reward,
                            mask_out=True,
                            training=False,
                            record=args.capture_video)]
                    eval_env = gym.vector.SyncVectorEnv(eval_env)
                    obs, infos = eval_env.reset()
                    done = False
                    pvm_buffer = PVMBuffer(args.pvm_stack, (eval_env.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
                    while not done:
                        pvm_buffer.append(obs)
                        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
                        motor_actions_pack, sensory_actions_pack = actor.get_action(resize(torch.from_numpy(pvm_obs)).to(device))
                        motor_actions = motor_actions_pack["action"].detach().cpu().numpy()
                        sensory_actions = sensory_actions_pack["action"].detach().cpu().numpy()
                        next_obs, rewards, dones, _, infos = eval_env.step({"motor_action": motor_actions, 
                                                "sensory_action": [sensory_action_set[a] for a in  sensory_actions] })
                        obs = next_obs
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
                                    torch.save({"sfn": sfn.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict(), "actor": actor.state_dict()}, os.path.join(model_file_dir, model_fn))

                writer.add_scalar("charts/eval_episodic_return", np.mean(eval_episodic_returns), global_transitions)
                writer.add_scalar("charts/eval_episodic_return_std", np.std(eval_episodic_returns), global_transitions)
                # writer.add_scalar("charts/eval_episodic_length", np.mean(), global_transitions)
                print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}] [R list: {','.join([str(r) for r in eval_episodic_returns])}]")

                qf1.train()
                qf2.train()
                actor.train()
                sfn.train()

        obs = obs_backup # restore obs if eval occurs

    envs.close()
    eval_env.close()
    writer.close()