from gym.envs.registration import register

register(
    id='ss2d-v0',
    entry_point='ss2d.envs:SS2D_env',
)

