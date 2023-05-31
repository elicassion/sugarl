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
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torchvision.transforms import Resize

from common.buffer import ReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything
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
    parser.add_argument("--update-frequency", type=int, default=4,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
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
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        if isinstance(envs.single_action_space, Discrete):
            action_space_size = envs.single_action_space.n
        elif isinstance(envs.single_action_space, Dict):
            action_space_size = envs.single_action_space["motor_action"].n
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, action_space_size))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        if isinstance(envs.single_action_space, Discrete):
            action_space_size = envs.single_action_space.n
        elif isinstance(envs.single_action_space, Dict):
            action_space_size = envs.single_action_space["motor_action"].n
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, action_space_size))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


if __name__ == "__main__":
    args = parse_args()
    args.env = args.env.lower()
    args.capture_video = False
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

    resize = Resize((84, 84))

    # get a discrete observ action space
    OBSERVATION_SIZE = (84, 84)
    observ_x_max, observ_y_max = OBSERVATION_SIZE[0]-args.fov_size, OBSERVATION_SIZE[1]-args.fov_size
    sensory_action_step = (observ_x_max//args.sensory_action_x_size,
                          observ_y_max//args.sensory_action_y_size)
    sensory_action_x_set = list(range(0, observ_x_max, sensory_action_step[0]))[:args.sensory_action_x_size]
    sensory_action_y_set = list(range(0, observ_y_max, sensory_action_step[1]))[:args.sensory_action_y_size]
    sensory_action_set = list(product(sensory_action_x_set, sensory_action_y_set))

    for i in range(0, len(sensory_action_set), args.sensory_action_y_size):
        if (i // args.sensory_action_y_size) % 2 == 1:
            sensory_action_set[i:i+args.sensory_action_y_size] = sensory_action_set[i+args.sensory_action_y_size-1:i-1:-1]

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space["motor_action"].n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space["motor_action"],
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
            sensory_actions = sensory_action_set[random.randint(0, len(sensory_action_set)-1)]
        else:
            actions, _, _ = actor.get_action(torch.Tensor(pvm_obs).to(device))
            actions = actions.detach().cpu().numpy()
            motor_actions = actions
            sensory_actions = sensory_action_set[random.randint(0, len(sensory_action_set)-1)]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step({"motor_action": motor_actions, 
                        "sensory_action": [sensory_actions] })

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for idx, d in enumerate(dones):
                if d:
                    print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [R: {infos['final_info'][idx]['reward']:.2f}]")
                    writer.add_scalar("charts/episodic_return", infos['final_info'][idx]["reward"], global_transitions)
                    writer.add_scalar("charts/episodic_length", infos['final_info'][idx]["ep_len"], global_transitions)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
                fov_idx = random.randint(0, len(sensory_action_set)-1)
        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(real_next_obs)
        real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
        rb.add(pvm_obs, real_next_pvm_obs, motor_actions, rewards, dones, {})

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # INC total transitions
        global_transitions += args.env_num

        # ALGO LOGIC: training.
        if global_transitions > args.learning_starts:
            if global_transitions % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_transitions % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_transitions % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_transitions)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_transitions)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_transitions)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_transitions)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_transitions)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_transitions)
                writer.add_scalar("losses/alpha", alpha, global_transitions)
                # print("SPS:", int(global_transitions / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_transitions / (time.time() - start_time)), global_transitions)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_transitions)

            # evaluation
            if (global_transitions % args.eval_frequency == 0 and args.eval_frequency > 0) or \
               (global_transitions >= args.total_timesteps):
                qf1.eval()
                qf2.eval()
                actor.eval()
                
                eval_episodic_returns, eval_episodic_lengths = [], []

                for eval_ep in range(args.eval_num):
                    eval_env = [make_env(args.env, args.seed+eval_ep, frame_stack=args.frame_stack, action_repeat=args.action_repeat, 
                            fov_size=(args.fov_size, args.fov_size), 
                            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                            sensory_action_mode=args.sensory_action_mode,
                            sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
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
                        actions, _, _ = actor.get_action(torch.Tensor(pvm_obs).to(device))
                        actions = actions.detach().cpu().numpy()
                        next_obs, rewards, dones, _, infos = envs.step({"motor_action": motor_actions, 
                                                                        "sensory_action": [sensory_action_set[random.randint(0, len(sensory_action_set)-1)]] })
                        obs = next_obs
                        done = dones[0]
                        if done:
                            eval_episodic_returns.append(infos['final_info'][0]["reward"])
                            eval_episodic_lengths.append(infos['final_info'][0]["ep_len"])
                            fov_idx = random.randint(0, len(sensory_action_set)-1)
                            if args.capture_video:
                                record_file_dir = os.path.join("recordings", args.exp_name, os.path.basename(__file__).rstrip(".py"), args.env)
                                os.makedirs(record_file_dir, exist_ok=True)
                                record_file_fn = f"{args.env}_seed{args.seed}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                eval_env.envs[0].save_record_to_file(os.path.join(record_file_dir, record_file_fn))
                                model_file_dir = os.path.join("trained_models", args.exp_name, os.path.basename(__file__).rstrip(".py"), args.env)
                                os.makedirs(model_file_dir, exist_ok=True)
                                model_fn = f"{args.env}_seed{args.seed}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                                torch.save({"sfn": None, "qf1": qf1.state_dict(), "qf2": qf2.state_dict(), "actor": actor.state_dict()}, os.path.join(model_file_dir, model_fn))

                writer.add_scalar("charts/eval_episodic_return", np.mean(eval_episodic_returns), global_transitions)
                writer.add_scalar("charts/eval_episodic_return_std", np.std(eval_episodic_returns), global_transitions)
                # writer.add_scalar("charts/eval_episodic_length", np.mean(), global_transitions)
                print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}] [R list: {','.join([str(r) for r in eval_episodic_returns])}]")

                qf1.train()
                qf2.train()
                actor.train()

    envs.close()
    eval_env.close()
    writer.close()