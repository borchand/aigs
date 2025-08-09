# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from pcgym.envs import PcgrlEnv
from typing import Tuple
from pcgym.envs.helper import get_string_map
from PIL import Image


# %% Init population (maps)
def init(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.qd.game, rep=cfg.qd.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.qd.n, *env._rep._map.shape))  # type: ignore
    return env, pop


# %% using gym-pcg to evaluate map quality
def eval(env, p) -> Tuple[int, dict]:
    env._rep._map = p
    string_map = get_string_map(env._rep._map, env._prob.get_tile_types())
    stats = env._prob.get_stats(string_map)
    return env._prob.get_reward(stats, {k: 0 for k in stats.keys()}), stats


def mutate(cfg, env, p) -> np.ndarray:
    mask = np.random.random(p.shape) < cfg.qd.p
    p[mask] = np.random.randint(0, env.get_num_tiles(), p[mask].shape)
    return p


# %% Setup
def main(ctx):
    env, pop = init(ctx.config)
    fitness, behavior = zip(*list(map(lambda p: eval(env, p), pop)))
    print(fitness)
    print(behavior)
    # print(list(map(lambda p: mutate(ctx.config, env, p), pop)))
