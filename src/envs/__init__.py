from functools import partial
from .multiagentenv import MultiAgentEnv

from .ssd import HarvestEnv,CleanupEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY['harvest'] = partial(env_fn,env=HarvestEnv)
REGISTRY['cleanup'] = partial(env_fn,env=CleanupEnv)

