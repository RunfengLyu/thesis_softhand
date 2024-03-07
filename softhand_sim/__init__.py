import gymnasium 
from gymnasium.envs.registration import register
print("yes")
register(
    id="softhand_sim/SofthandTest-v0",
    entry_point="softhand_sim.envs:HandEnv",
)