from gym.envs.registration import registry, register, make, spec
import numpy as np

register(
    id='BaxterEnv-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=100,
    kwargs={'ball_loc': np.array([1.25, 0.4, 1.1]),
    	'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
    	#'active_joint_list':[False,False,False,True,True,True,False]
    	'active_joint_list':np.array([False,False,False,True,True,True,False]),
    	'random_start':True
    }
)