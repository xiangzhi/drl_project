#!/usr/bin/env python


from gym.envs.registration import registry, register, make, spec
import gym
import matplotlib.pyplot as plt
from deep.noise_generator import OU_Generator
import numpy as np


def test_OU_noise():
    gen = OU_Generator(np.zeros(3),theta=0.1, sigma=0.05)
    arr = np.zeros((1000,3))
    curr_pose = np.array([-0.3,-0.3,-0.3])
    curr_pose = np.array([-1,-1,-1])
    for i in range(0,1000):
        noise = (gen.generate(curr_pose,i))
        curr_pose = curr_pose + noise
        arr[i] = curr_pose
    plt.plot(arr)
    plt.show()

if __name__ == '__main__':
	test_OU_noise()

# register(
# 	id='BaxterEnv-v0',
# 	entry_point='baxter_env:BaxterEnv',
# 	max_episode_steps=10,
# )


# if __name__ == '__main__':
# 	env = gym.make('BaxterEnv-v0')
# 	(joints, image) = env.reset()
# 	plt.ion()
# 	plt.imshow(image)
# 	plt.show()
# 	plt.pause(0.0001)
# 	for x in range(0,10):
# 		state = env.reset()
# 		for i in range(0,10):
			
# 			plt.imshow(state[1])
# 			plt.pause(0.0001)
# 			#plt.draw()
# 			action = env.action_space.sample()
# 			#print(state)
# 			state, reward, terminal, info = env.step(action)
# 			print(reward)



