import gym
import ptan.ptan as ptan
import argparse
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

import common
from dqn_model import RainbowDQN, PrioReplayBuffer

NAME = "combined_ext"
N_STEPS = 4
PRIO_REPLAY_ALPHA = 0.6


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--envs", type=int, default=3, help="Amount of environments to run in parallel")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = []
    for _ in range(args.envs):
        env = gym.make(params.env_name)
        env = ptan.common.wrappers.wrap_dqn(env)
        env.seed(common.SEED)
        envs.append(env)

    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    params.batch_size *= args.envs
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
                                                           gamma=params.gamma,
                                                           steps_count=N_STEPS)
    buffer = PrioReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss_v, sample_prios = common.calc_loss_prio(batch, batch_weights,
                                                     net, tgt_net.target_model,
                                                     gamma=params.gamma**N_STEPS,
                                                     device=device)
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "beta": buffer.update_beta(engine.state.iteration),
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME, args.envs, ['avg_loss', 'avg_fps'])
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size, args.envs))





