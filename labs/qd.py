# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_int_prob, get_string_map
from test.test_buffer import flatten
from PIL import Image
from tqdm import tqdm

# Booth test function
@partial(np.vectorize, signature="(d)->()")
def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(pop**2) / 4000 - np.prod(np.cos(pop / np.sqrt(np.arange(1, pop.size + 1))))


@partial(np.vectorize, signature="(d)->(d)", excluded=[0])
def mutate(sigma, pop):  # What are we doing here?
    return pop + np.random.normal(0, sigma, pop.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # TODO: think about what we are doing here. Is it smart?
    return x1 * np.random.rand() + x2 * (1 - np.random.rand())


def step_griewank(pop, cfg):
    loss = griewank_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
    return mutate(cfg.sigma, pop), loss  # return new generation and loss

def step_rastrigin(pop, cfg):
    loss = rastrigin_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
    return mutate(cfg.sigma, pop), loss  # return new generation and loss

def evolutionary_algo(cfg):
    x = np.random.rand(cfg.population, cfg.dimensions)

    results = []
    for i in range(cfg.generation):
        x, loss = step_rastrigin(x, cfg)
        print(f"Generation {i} best loss: {loss.min()}")
        results.append((x, loss))

    # plot loss per generation
    plt.plot([r[1].min() for r in results])
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.title("Rastrigin Function Optimization")
    plt.show()

    plt.savefig("rastrigin_optimization.png")

    all_xs = np.array([r[0] for r in results])[0]
    print(all_xs.shape)
    r = np.array([rastrigin_function(x) for x in all_xs])
    print(r.shape)

    scatter = plt.scatter(all_xs[:, 0], all_xs[:, 1], c=r, cmap='viridis')
    plt.colorbar(scatter, label="Loss")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rastrigin Function Landscape")
    plt.show()

    plt.savefig("rastrigin_landscape.png")

def random_map(cfg):
    env, pop = init_pcgym(cfg)
    map = get_string_map(env._rep._map, env._prob.get_tile_types())
    behavior = env._prob.get_stats(map)
    print(behavior)
    exit()


# %% Setup
def main(cfg):
    print(f"Running {cfg.run}")
    if cfg.run == "random":
        random_map(cfg)
    elif cfg.run == "ea":
        evolutionary_algo(cfg)
    elif cfg.run == "pcgym":
        env, pop = init_pcgym(cfg)
        I = 100000
        G = min(I * 0.1, len(pop))

        archive = {}

        for i in tqdm(range(I)):
            if i < G:
                # random solution
                xt = pop[i]
            else:
                # random selection
                keys = list(archive.keys())
                x1 = archive[keys[np.random.randint(0, len(keys))]]["solution"]
                x2 = archive[keys[np.random.randint(0, len(keys))]]["solution"]

                xt = mutate(.5, crossover(x1, x2))

            stats = get_stats(env, xt)
            n_jumps, n_enemies = get_behavior(stats)

            fit = fitness(stats, env)

            k = get_key(n_jumps, n_enemies)
            if k not in archive or archive[k]["fitness"] < fit:
                archive[k] = {"solution" : xt, "fitness": fit}

        # plot archive
        xs = np.array([k[0] for k in archive.keys()])
        ys = np.array([k[1] for k in archive.keys()])
        fitnesses = np.array([v["fitness"] for v in archive.values()])
        plt.scatter(xs, ys, c=fitnesses, cmap="viridis")
        plt.colorbar(label="Fitness")
        plt.xlabel("Number of Jumps")
        plt.ylabel("Number of Enemies")
        plt.title("Archive of Solutions")
        plt.savefig("archive.png")

        # generate image of best solution
        best = max(archive.values(), key=lambda x: x["fitness"])["solution"]
        env._rep._map = best
        env.reset()
        Image.fromarray(obj=env.render()).save(fp="best_solution.png")




# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop


def fitness(stats, env):
    map_size = env._rep._map.shape[0] * env._rep._map.shape[1]
    map_width = env._rep._map.shape[1]
    return ((map_width - stats["dist-win"]) / map_width) + (stats["empty"] / map_size)


def get_key(n_jumps, n_enemies):
    return (n_jumps, n_enemies // 40 * 40)



def get_stats(env, xt):
    tile_types = env._prob.get_tile_types()
    x_str = [[tile_types[min(int(tile_int), len(tile_types) - 1)] for tile_int in row] for row in xt]

    return env._prob.get_stats(x_str)

def get_behavior(stats):
    return stats["jumps"], stats["enemies"]
