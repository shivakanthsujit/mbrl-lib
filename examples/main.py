# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import hydra
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import yaml

import mbrl.algorithms.planet as planet
import mbrl.util.env
from dataclasses import dataclass

from pclast_envs.env_wrapper import make_env
from resize_obs import ResizeObservation
from termcolor import colored 

@dataclass
class TrainConfig:
    env_id: str = "polygon-obs"
    random_start: bool = True
    random_goal: bool = True
    timelimit: int = 100
    seed: int = 0
    nrandom_goals: int = 32
    sparse_reward: bool = False
    lns_reward: bool = False

env_cfg = TrainConfig()

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    # env, _, _ = mbrl.util.env.EnvHandler.make_env(cfg)
    
    assert isinstance(cfg.sparse_reward, bool)
    if isinstance(cfg.sparse_reward, str):
        if sparse_reward == "True":
            sparse_reward = True
        elif sparse_reward == "False":
            sparse_reward = False
    
    env_cfg.env_id = cfg.env_name
    env_cfg.sparse_reward = cfg.sparse_reward
    print(colored(f"{env_cfg.env_id}, {env_cfg.sparse_reward}", "red"))
    env = make_env(env_cfg, timelimit=cfg.overrides.trial_length)
    env = ResizeObservation(env, (64, 64))

    low = env.action_space.low.tolist()
    high = env.action_space.high.tolist()
    action_dim = env.action_space.shape

    # cfg.action_optimizer.lower_bound = low
    # cfg.action_optimizer.upper_bound = high

    # cfg.algorithm.agent.action_lb = low
    # cfg.algorithm.agent.action_ub = high

    # cfg.algorithm.agent.optimizer_cfg.lower_bound = low
    # cfg.algorithm.agent.optimizer_cfg.upper_bound = high

    # cfg.dynamics_model.action_size = action_dim

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    planet_model, agent, _ = planet.train(env, cfg)

    rews = []
    for _ in tqdm(range(cfg.eval_eps), desc="Eval Loop"):
        # Collect one episode of data
        episode_reward = 0.0
        obs = env.reset()
        agent.reset()
        planet_model.reset_posterior()
        action = None
        done = False
        while not done:
            planet_model.update_posterior(obs, action=action)
            action_noise = 0
            action = agent.act(obs) + action_noise
            action = np.clip(action, -1.0, 1.0)  # to account for the noise
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        rews.append(episode_reward)

    rews = np.array(rews)
    mean = rews.mean()
    std = rews.std()
    fname = os.path.join(os.getcwd(), "eval.npz")
    np.savez(fname, rews=rews)
    print(f"Eval saved to {fname}")
    print(f"{mean:.2f} +- {std:.2f}")



if __name__ == "__main__":
    run()
