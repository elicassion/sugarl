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
from gymnasium.spaces import Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter

from common.buffer import NstepRewardDoubleActionReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import (
    get_sugarl_reward_scale_dmc,
    get_timestr, 
    schedule_drq,
    seed_everything,
    soft_update_params,
    weight_init_drq,
    TruncatedNormal
)

from active_gym import DMCFixedFovealEnv, DMCEnvArgs


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances")
    
    # env setting
    parser.add_argument("--domain-name", type=str, default="walker",
        help="the name of the dmc domain")
    parser.add_argument("--task-name", type=str, default="walk",
        help="the name of the dmc task")
    parser.add_argument("--env-num", type=int, default=1, 
        help="# envs in parallel")
    parser.add_argument("--frame-stack", type=int, default=3,
        help="frame stack #")
    parser.add_argument("--action-repeat", type=int, default=2,
        help="action repeat #") # i.e. frame skip
    parser.add_argument("--clip-reward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True) # dmc we may not clip

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
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2000,
        help="timestep to start learning")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="the learning rate of drq")
    parser.add_argument("--update-frequency", type=int, default=2,
        help="update frequency of drq")
    parser.add_argument("--stddev-clip", type=float, default=0.3)
    parser.add_argument("--stddev-schedule", type=str, default="linear(1.0,0.1,50000)")
    parser.add_argument("--feature-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-step-reward", type=int, default=3)
    
    # eval args
    parser.add_argument("--eval-frequency", type=int, default=-1,
        help="eval frequency. default -1 is eval at the end.")
    parser.add_argument("--eval-num", type=int, default=10,
        help="eval episodes")
    
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(domain_name, task_name, seed, **kwargs):
    def thunk():
        env_args = DMCEnvArgs(
            domain_name=domain_name, task_name=task_name, seed=seed, obs_size=(84, 84), **kwargs
        )
        env = DMCFixedFovealEnv(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init_drq)

    def forward(self, obs):
        obs = obs - 0.5 # /255 is done by env
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class ActiveActor(nn.Module):
    def __init__(self, 
                 repr_dim, 
                 motor_action_shape, 
                 sensory_action_shape, 
                 feature_dim, 
                 hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.motor_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, motor_action_shape[0]))
        
        self.sensory_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, sensory_action_shape[0]))

        self.apply(weight_init_drq)

    def forward(self, obs, std):
        h = self.trunk(obs)

        motor_mu = self.motor_policy(h)
        motor_mu = torch.tanh(motor_mu)
        motor_std = torch.ones_like(motor_mu) * std
        motor_dist = TruncatedNormal(motor_mu, motor_std)

        sensory_mu = self.sensory_policy(h)
        sensory_mu = torch.tanh(sensory_mu)
        sensory_std = torch.ones_like(sensory_mu) * std
        sensory_dist = TruncatedNormal(sensory_mu, sensory_std)
        return motor_dist, sensory_dist


class ActiveCritic(nn.Module):
    def __init__(self, repr_dim, motor_action_shape, sensory_action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.motor_q1 = nn.Sequential(
            nn.Linear(feature_dim + motor_action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.motor_q2 = nn.Sequential(
            nn.Linear(feature_dim + motor_action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        
        self.sensory_q1 = nn.Sequential(
            nn.Linear(feature_dim + sensory_action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.sensory_q2 = nn.Sequential(
            nn.Linear(feature_dim + sensory_action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(weight_init_drq)

    def forward(self, obs, motor_action, sensory_action):
        h = self.trunk(obs)
        h_motor_action = torch.cat([h, motor_action], dim=-1)
        motor_q1 = self.motor_q1(h_motor_action)
        motor_q2 = self.motor_q2(h_motor_action)

        h_sensory_action = torch.cat([h, sensory_action], dim=-1)
        sensory_q1 = self.sensory_q1(h_sensory_action)
        sensory_q2 = self.sensory_q2(h_sensory_action)

        return motor_q1, motor_q2, sensory_q1, sensory_q2
    
class SelfPredictionNetwork(nn.Module):
    def __init__(self, repr_dim, motor_action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim*2, feature_dim*2),
                                   nn.LayerNorm(feature_dim*2), nn.Tanh())
        self.pred_net = nn.Sequential(nn.Linear(feature_dim*2, hidden_dim),
                                      nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(inplace=True), nn.Linear(hidden_dim, motor_action_shape[0]))

    def forward(self, obs, next_obs):
        obs = torch.cat([obs, next_obs], dim=1)
        action_mu = self.pred_net(self.trunk(obs))
        action_mu = torch.tanh(action_mu)
        return action_mu



class DrQV2SugarlAgent:
    def __init__(self, obs_shape, motor_action_shape, sensory_action_shape, 
                 device, lr, feature_dim,
                 hidden_dim, critic_target_tau, learning_starts,
                 update_every_steps, stddev_schedule, stddev_clip,
                 sensory_action_step, sugarl_reward_scale):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.learning_starts = learning_starts
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.sensory_action_step = sensory_action_step # for dicretize sensory action
        self.sugarl_reward_scale = sugarl_reward_scale

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = ActiveActor(self.encoder.repr_dim, motor_action_shape, sensory_action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = ActiveCritic(self.encoder.repr_dim, motor_action_shape, sensory_action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = ActiveCritic(self.encoder.repr_dim, motor_action_shape, sensory_action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.sfn = SelfPredictionNetwork(self.encoder.repr_dim, motor_action_shape, feature_dim, hidden_dim).to(device)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.sfn_opt = torch.optim.Adam(self.sfn.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.sfn.train(training)
    
    def eval(self):
        self.train(False)

    def get_discretized_sensory_action(self, va: np.ndarray, thresh=0.5)-> np.ndarray:
        threshed = np.zeros_like(va, dtype=np.int32)
        threshed [np.where((va >= -thresh) & (va <= thresh))] = 0
        threshed [va<-thresh] = -1
        threshed [va>thresh] = 1

        threshed *= np.array(self.sensory_action_step, dtype=np.int32)
        return threshed


    def act(self, obs, step, eval_mode=None):
        obs = self.encoder(obs)
        stddev = schedule_drq(self.stddev_schedule, step)
        motor_dist, sensory_dist = self.actor(obs, stddev)

        # auto eval
        if eval_mode is None:
            eval_mode = not self.training
        
        if eval_mode:
            motor_action = motor_dist.mean
            sensory_action = sensory_dist.mean
        else:
            motor_action = motor_dist.sample(clip=None)
            if step < self.learning_starts:
                motor_action.uniform_(-1.0, 1.0)
            
            sensory_action = sensory_dist.sample(clip=None)
            if step < self.learning_starts:
                sensory_action.uniform_(-1.0, 1.0)
            
        motor_action = motor_action.detach().cpu().numpy()
        sensory_action = sensory_action.detach().cpu().numpy()
        
        processed_action = {
            "motor_action": motor_action,
            "sensory_action": self.get_discretized_sensory_action(sensory_action)
        }
        return motor_action, sensory_action, processed_action

    def update_critic(self, obs, motor_action, sensory_action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule_drq(self.stddev_schedule, step)
            motor_dist, sensory_dist = self.actor(next_obs, stddev)
            next_motor_action = motor_dist.sample(clip=self.stddev_clip)
            next_sensory_action = sensory_dist.sample(clip=self.stddev_clip)
            motor_target_Q1, motor_target_Q2, sensory_target_Q1, sensory_target_Q2 = self.critic_target(next_obs, 
                                                                                                            next_motor_action,
                                                                                                            next_sensory_action)
            motor_target_V = torch.min(motor_target_Q1, motor_target_Q2)
            sensory_target_V = torch.min(sensory_target_Q1, sensory_target_Q2)
            sum_target_V = motor_target_V + sensory_target_V
            sum_target_Q = reward + (discount * sum_target_V)

        motor_q1, motor_q2, sensory_q1, sensory_q2 = self.critic(obs, motor_action, sensory_action)
        critic_loss = F.mse_loss(motor_q1+sensory_q1, sum_target_Q) + \
                        F.mse_loss(motor_q2+sensory_q2, sum_target_Q)

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = schedule_drq(self.stddev_schedule, step)
        motor_dist, sensory_dist = self.actor(obs, stddev)
        motor_action = motor_dist.sample(clip=self.stddev_clip)
        motor_log_prob = motor_dist.log_prob(motor_action).sum(-1, keepdim=True)
        sensory_action = sensory_dist.sample(clip=self.stddev_clip)
        sensory_log_prob = sensory_dist.log_prob(sensory_action).sum(-1, keepdim=True)
        motor_q1, motor_q2, sensory_q1, sensory_q2 = self.critic(obs, motor_action, sensory_action)
        motor_q = torch.min(motor_q1, motor_q2)
        sensory_q = torch.min(sensory_q1, sensory_q2)

        actor_loss = - motor_q.mean() - sensory_q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return metrics
    
    def update_sfn(self, obs, next_obs, action, step):
        pred_action = self.sfn(obs, next_obs)
        sugarl_reward = - (pred_action.detach() - action).pow(2).sum(dim=-1) / action.size(1) # norm by action dim
        sfn_loss = F.mse_loss(pred_action, action)
        self.sfn_opt.zero_grad(set_to_none=True)
        sfn_loss.backward()
        self.sfn_opt.step()
        return sugarl_reward

    def update_sugarl_reward(self, reward, sugarl_reward):
        reward = reward + sugarl_reward * self.sugarl_reward_scale
        return reward

    def update(self, batch, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs = batch.observations
        motor_action = batch.motor_actions
        sensory_action = batch.sensory_actions
        next_obs = batch.next_observations
        reward = batch.rewards
        discount = batch.discounts

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        sugarl_reward = self.update_sfn(obs.detach(), next_obs, motor_action, step)
        reward = self.update_sugarl_reward(reward, sugarl_reward)

        self.update_critic(obs, motor_action, sensory_action, reward, discount, next_obs, step)

        self.update_actor(obs.detach(), step)

        # update critic target
        soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
    
    def save_agent(self):
        agent = {
            "encoder": self.encoder.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
            "sfn": self.sfn.state_dict(),
        }
        return agent


if __name__ == "__main__":
    args = parse_args()
    args.domain_name = args.domain_name.lower()
    args.task_name = args.task_name.lower()
    run_name = f"{args.domain_name}-{args.task_name}__{os.path.basename(__file__)}__{args.seed}__{get_timestr()}"
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
    # env setup
    envs = []
    for i in range(args.env_num):
        envs.append(make_env(args.domain_name, 
                             args.task_name,
                             args.seed+i, 
                             frame_stack=args.frame_stack, 
                             action_repeat=args.action_repeat,
                             fov_size=(args.fov_size, args.fov_size), 
                             fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                             sensory_action_mode=args.sensory_action_mode,
                             sensory_action_space=(-max(sensory_action_step), max(sensory_action_step)),
                             resize_to_full=args.resize_to_full,
                             mask_out=True,
                             clip_reward=False))
    envs = gym.vector.SyncVectorEnv(envs)

    sugarl_reward_scale = get_sugarl_reward_scale_dmc(args.domain_name, args.task_name)

    drq_agent = DrQV2SugarlAgent(
        envs.single_observation_space.shape,
        envs.single_action_space["motor_action"].shape,
        Box(low=-1., high=1., shape=(2,)).shape,
        device=device,
        lr=args.lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        critic_target_tau=args.tau,
        learning_starts=args.learning_starts,
        update_every_steps=args.update_frequency,
        stddev_schedule=args.stddev_schedule,
        stddev_clip=args.stddev_clip,
        sensory_action_step=sensory_action_step,
        sugarl_reward_scale=sugarl_reward_scale
    )

    rb = NstepRewardDoubleActionReplayBuffer(
        n_step_reward=args.n_step_reward,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        motor_action_space=envs.single_action_space["motor_action"],
        sensory_action_space=Box(low=-1., high=1., shape=(2,)),
        device=device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset()
    pvm_buffer = PVMBuffer(args.pvm_stack, (envs.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
    reset_pvm_buffer = False
    global_transitions = 0
    while global_transitions < args.total_timesteps:
        pvm_buffer.append(obs)
        pvm_obs = pvm_buffer.get_obs(mode="stack_max")
        motor_actions, sensory_actions, processed_actions = drq_agent.act(torch.Tensor(pvm_obs).to(device), global_transitions)
        next_obs, rewards, dones, _, infos = envs.step(processed_actions)

        if "final_info" in infos:
            for idx, d in enumerate(dones):
                if d:
                    print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [R: {infos['final_info'][idx]['reward']:.2f}]")
                    writer.add_scalar("charts/episodic_return", infos['final_info'][idx]["reward"], global_transitions)
                    writer.add_scalar("charts/episodic_length", infos['final_info'][idx]["ep_len"], global_transitions)
                    break

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
                reset_pvm_buffer = True

        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(real_next_obs)
        real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
        rb.add(pvm_obs, real_next_pvm_obs, motor_actions, sensory_actions, rewards, dones, [{}])

        if reset_pvm_buffer:
            pvm_buffer = PVMBuffer(args.pvm_stack, (envs.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
            reset_pvm_buffer = False


        obs = next_obs
        global_transitions += args.env_num

        # ALGO LOGIC: training.
        if global_transitions > args.learning_starts:
            data = rb.sample(args.batch_size)
            drq_agent.update(data, global_transitions)

        if global_transitions % 100 == 0:
            writer.add_scalar("charts/SPS", int(global_transitions / (time.time() - start_time)), global_transitions)

        if (global_transitions % args.eval_frequency == 0 and args.eval_frequency > 0) or \
            (global_transitions >= args.total_timesteps):
            drq_agent.eval()
            
            eval_episodic_returns, eval_episodic_lengths = [], []
            for eval_ep in range(args.eval_num):
                eval_env=[make_env(args.domain_name, 
                            args.task_name,
                            args.seed+eval_ep, 
                            frame_stack=args.frame_stack, 
                            action_repeat=args.action_repeat,
                            fov_size=(args.fov_size, args.fov_size), 
                            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                            sensory_action_mode=args.sensory_action_mode,
                            sensory_action_space=(-max(sensory_action_step), max(sensory_action_step)),
                            resize_to_full=args.resize_to_full,
                            mask_out=True,
                            clip_reward=False)]
                eval_env = gym.vector.SyncVectorEnv(eval_env)
                eval_obs, infos = eval_env.reset()
                eval_pvm_buffer = PVMBuffer(args.pvm_stack, (envs.num_envs, args.frame_stack,)+OBSERVATION_SIZE)
                done = False
                while not done:
                    eval_pvm_buffer.append(eval_obs)
                    eval_pvm_obs = eval_pvm_buffer.get_obs(mode="stack_max")
                    motor_actions, sensory_actions, processed_actions = drq_agent.act(torch.Tensor(eval_pvm_obs).to(device), step=global_transitions)
                    eval_next_obs, rewards, dones, _, infos = eval_env.step(processed_actions)
                    eval_obs = eval_next_obs
                    done = dones[0]
                    if done:
                        eval_episodic_returns.append(infos['final_info'][0]["reward"])
                        eval_episodic_lengths.append(infos['final_info'][0]["ep_len"])
                        if args.capture_video:
                            record_file_dir = os.path.join("recordings", args.exp_name, os.path.basename(__file__).replace(".py", ""), f"{args.domain_name}-{args.task_name}")
                            os.makedirs(record_file_dir, exist_ok=True)
                            record_file_fn = f"{args.domain_name}-{args.task_name}_seed{args.seed}_step{global_transitions:07d}_eval{eval_ep:02d}_record.pt"
                            eval_env.envs[0].save_record_to_file(os.path.join(record_file_dir, record_file_fn))
                            if global_transitions >= args.total_timesteps and eval_ep == 0:
                                model_file_dir = os.path.join("trained_models", args.exp_name, os.path.basename(__file__).replace(".py", ""), f"{args.domain_name}-{args.task_name}")
                                os.makedirs(model_file_dir, exist_ok=True)
                                model_fn = f"{args.domain_name}-{args.task_name}_seed{args.seed}_model.pt"
                                torch.save({"sfn": None, "agent": drq_agent.save_agent()}, os.path.join(model_file_dir, model_fn))

            writer.add_scalar("charts/eval_episodic_return", np.mean(eval_episodic_returns), global_transitions)
            writer.add_scalar("charts/eval_episodic_return_std", np.std(eval_episodic_returns), global_transitions)
            # writer.add_scalar("charts/eval_episodic_length", np.mean(), global_transitions)
            print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}] [R list: {','.join([str(r) for r in eval_episodic_returns])}]")

            drq_agent.train()

    envs.close()
    writer.close()