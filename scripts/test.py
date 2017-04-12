#!/usr/bin/env python


from gym.envs.registration import registry, register, make, spec
import gym
import matplotlib.pyplot as plt

register(
	id='BaxterEnv-v0',
	entry_point='baxter_env:BaxterEnv',
	max_episode_steps=10,
)


if __name__ == '__main__':
	env = gym.make('BaxterEnv-v0')
	(joints, image) = env.reset()
	plt.ion()
	plt.imshow(image)
	plt.show()
	plt.pause(0.0001)
	for x in range(0,10):
		state = env.reset()
		for i in range(0,10):
			
			plt.imshow(state[1])
			plt.pause(0.0001)
			#plt.draw()
			action = env.action_space.sample()
			#print(state)
			state, reward, terminal, info = env.step(action)
			print(reward)



