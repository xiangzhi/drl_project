#!/usr/bin/env python


from gym.envs.registration import registry, register, make, spec
import gym

register(
	id='BaxterEnv-v0',
	entry_point='baxter_env:BaxterEnv',
	max_episode_steps=200,
)


if __name__ == '__main__':
	env = gym.make('BaxterEnv-v0')
	state = env.reset()


	for i in range(0,100):
		print(state)
		action = [0.2,-0.2,0,0,0,0,0]
		state, reward, terminal, info = env.step(action)



