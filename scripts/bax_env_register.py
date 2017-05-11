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
        'random_start':False,
        'neg_reward':-0.1
    }
)

register(
    id='BaxterEnv-v1',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=100,
    kwargs={'ball_loc': np.array([1.25, 0.4, 1.1]),
        'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
        #'active_joint_list':[False,False,False,True,True,True,False]
        'active_joint_list':np.array([False,False,False,True,True,True,False]),
        'random_start':True,
        'neg_reward':-0.1
    }
)

register(
    id='BaxterEnv-v2',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=100,
    kwargs={
        'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
        #'active_joint_list':[False,False,False,True,True,True,False]
        'active_joint_list':np.array([False,False,False,True,True,True,False]),
        'random_start':True,
        'neg_reward':-0.1
    }
)

register(
    id='BaxterEnv-v3',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=100,
    kwargs={
        'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
        #'active_joint_list':[False,False,False,True,True,True,False]
        'active_joint_list':np.array([False,False,True,True,True,True,False]),
        'random_start':True,
        'neg_reward':-0.1
    }
)

register(
    id='BaxterEnv-v4',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=100,
    kwargs={
        'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
        #'active_joint_list':[False,False,False,True,True,True,False]
        'active_joint_list':np.array([True,True,True,True,True,True,False]),
        'random_start':True,
        'neg_reward':-0.1
    }
)

register(
    id='BaxterEnv-joints-only-v0',
    entry_point='baxter_env:BaxterEnv',
    max_episode_steps=100,
    kwargs={'ball_loc': np.array([1.25, 0.4, 1.1]),
        'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
        #'active_joint_list':[False,False,False,True,True,True,False]
        'active_joint_list':np.array([False,False,False,True,True,True,False]),
        'random_start':False,
        'joint_only':True,
        'neg_reward':-0.1
    }
)


# register(
#     id='BaxterEnv-gravity-joints-only-v0',
#     entry_point='baxter_env:BaxterEnvGravity',
#     max_episode_steps=100,
#     kwargs={'ball_loc': np.array([1.25, 0.4, 1.1]),
#         'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
#         #'active_joint_list':[False,False,False,True,True,True,False]
#         'active_joint_list':np.array([False,False,False,True,True,True,False]),
#         'random_start':False,
#         'joint_only':True,
#         'neg_reward':-0.1
#     }
# )

# register(
#     id='BaxterEnv-gravity-joints-only-v1',
#     entry_point='baxter_env:BaxterEnvGravity',
#     max_episode_steps=100,
#     kwargs={'ball_loc': np.array([1.25, 0.4, 1.1]),
#         'start_joint_list': np.array([-0.7,-0.75,0,2, 0,-1.35, 0]),
#         #'active_joint_list':[False,False,False,True,True,True,False]
#         'active_joint_list':np.array([False,False,False,True,True,True,False]),
#         'random_start':True,
#         'joint_only':True,
#         'neg_reward':-0.1
#     }
# )