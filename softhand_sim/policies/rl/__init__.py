from .td3 import TD3
from .sac import SAC


RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
}