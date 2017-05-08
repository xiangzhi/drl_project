from gym.envs.registration import registry, register, make, spec

register(
    id='BaxterEnv-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=50,
    kwargs={'ball_loc': np.array([2.56, 3.])})
)