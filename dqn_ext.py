import gym
import ptan.ptan as ptan
import argparse
import random

import torch
import torch.optim as optim

from ignite.engine import Engine
import common
from dqn_model import DQN

NAME = "02_n_steps"
DEFAULT_N_STEPS = 3

if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", type=int, default=DEFAULT_N_STEPS,
                        help="Steps to do on Bellman unroll")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent,
                                                           gamma=params.gamma,
                                                           steps_count=args.n)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        # made an additional variable, because in the original training falls
        gamm = params.gamma**args.n
        loss_v = common.clac_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=gamm,
                                      device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, f"{NAME}={args.n}")
    engine.run(common.batch_generator(buffer, params.replay_initial,
                                      params.batch_size))




