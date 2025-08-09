# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from pcgym.envs import PcgrlEnv
from typing import Tuple
from pcgym.envs.helper import get_string_map


def init(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob="smb", rep="turtle", render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.args.n, *env._rep._map.shape))  # type: ignore
    return env, pop


def eval(env, p) -> int:
    env._rep._map = p
    string_map = get_string_map(env._rep._map, env._prob.get_tile_types())
    stats = env._prob.get_stats(string_map)
    baseline = {k: 0 for k in stats.keys()}
    quality = env._prob.get_reward(stats, baseline)
    return quality


# %% Setup
def main(cfg):
    env, pop = init(cfg)
    print(list(map(lambda p: eval(env, p), pop)))
