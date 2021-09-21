from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import gym
from utils import AverageMeter
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Play CartPole')
    parser.add_argument('--num-episodes', default=100, type=int, help='Number of episodes')
    parser.add_argument('--episode-length', default=10000, type=int, help='Maximum length of episodes')
    parser.add_argument('--render', action='store_true', help='Render the game')
    parser.add_argument('--policy', default='theta_omega', type=str, 
                        help='Choose the hard coded policy: random, theta, omega or theta_omega')

    return parser.parse_args()

def env_reset(env, max_eps=300):
    obs = env.reset()
    env._max_episode_steps = max_eps

    return obs

def theta_policy(obs):
    theta = obs[2]
    return 0 if theta < 0 else 1

def omega_policy(obs):
    w = obs[3]
    return 0 if w < 0 else 1

def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

if __name__=='__main__':
    args = parse_args()

    env = gym.make('CartPole-v0')
    rewards_meter = AverageMeter()
    for i_episode in range(args.num_episodes):
        obs = env_reset(env, args.episode_length)
        for t in range(args.episode_length):
            if args.render:
                env.render()

            if args.policy == 'random':
                action = env.action_space.sample()
            elif args.policy == 'theta':
                action = theta_policy(obs)
            elif args.policy == 'omega':
                action = omega_policy(obs)
            elif args.policy == 'theta_omega':
                action = theta_omega_policy(obs)
            else:
                raise Exception(f'Unsupported policy: {args.policy}')

            obs, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                writer.add_scalar('Cum Rewards', t+1, i_episode)

                rewards_meter.update(t+1)
                writer.add_scalar('Average Rewards', rewards_meter.avg, i_episode)
                break
    env.close()