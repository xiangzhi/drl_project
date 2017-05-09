import gym
import rospy
import time
import numpy as np

import baxter_interface
from std_srvs.srv import Empty
from std_msgs.msg import Bool

from gazebo_msgs.srv import SetModelConfiguration, SetLinkProperties, GetLinkProperties, SetModelState 
from gazebo_msgs.msg import ModelState 
from sensor_msgs.msg import Image
from scipy.misc import imsave
from geometry_msgs.msg import Pose, Twist

import matplotlib.pyplot as plt

_max_limit = np.array([1,0.75,3,2.6,3,2,3])
_min_limit = np.array([-1.6,-2,-3,0,-3,-1.57,-3])


def generate_random_pose():
    """
    Generates a random baxter pose
    """
    return np.random.uniform(_min_limit, _max_limit)

class BaxterActionSpace(gym.Space):


    def __init__(self, joint_list):
        self._max_vel = np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3])
        self._min_vel = np.array([-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3])
        self.high = self._max_vel
        self.low = self._min_vel
        self._active_joint_list = joint_list
        self._num_joint = 7
        self.shape = (np.sum(joint_list),)
    def sample(self):
        """
        Randomly return a sample
        """

        #randomly generates the actions
        gen_velocities = np.random.uniform(self._min_vel, self._max_vel)

        #return the joints according to the active joint list
        return  gen_velocities[self._active_joint_list] 

    def clip_values(self, values):
        curr_action = np.clip(values,self.low[self._active_joint_list],self.high[self._active_joint_list])
        return actions

    def contains(self, x):

        #check if the stuff is in range

        x_full = np.zeros(self._num_joint)
        x_full[self._active_joint_list] = x

        return (x_full < self._max_vel).all() and (x_full > self._min_vel).all()




class BaxterEnv(gym.Env):


    def _image_callback(self, image):

        self._last_image = np.fromstring(image.data, dtype=np.uint8).reshape(800,800,3)
        #imsave('out.jpg',self._last_image)


    def __init__(self, start_joint_list=None, ball_loc=None, active_joint_list=None,random_start=False):

        rospy.init_node('baxter_env_v0',anonymous=True)

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        #services used by gazebo 
        self._pause_gazebo = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        self._unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self._reset_baxter = rospy.ServiceProxy('/gazebo/reset_world',Empty)
        self._reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        self._set_model = rospy.ServiceProxy('/gazebo/set_model_configuration',SetModelConfiguration)
        self._set_link = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self._get_link = rospy.ServiceProxy('/gazebo/get_link_properties',GetLinkProperties)
        rospy.Subscriber('/cameras/left_hand_camera/image',Image,self._image_callback)
        #publisher
        self._enable_pub = rospy.Publisher('/robot/set_super_enable',Bool,queue_size=1)

        self._ball_prop = self._get_link("ball::ball")

        if(ball_loc is not None):
            new_pose = Pose()
            new_pose.orientation.y = 0
            new_pose.position.x = ball_loc[0]
            new_pose.position.y = ball_loc[1]
            new_pose.position.z = ball_loc[2]
            self._ball_pose = new_pose
        else:
            self._ball_pose = None

        self._start_joint_list = start_joint_list
        self._random_start = random_start

        #active_joint_list which joints are active
        
        self._active_joint_list = active_joint_list
        if(active_joint_list is None):
            self._active_joint_list = np.array([True,True,True,True,True,True,True])

        self.action_space = BaxterActionSpace(self._active_joint_list)

        self.viewer = None

        self._num_joint = 7
        self._error_collector = np.zeros(7)
        self._neg_reward = -0.1


        self._total_reset()
        self._joint_angles = self._convert_joint_angle(self._left_arm.joint_angles())
        self._reset()
        

    def _calculate_reward(self, state):
        """
        calculate reward, right now is just how much green pixels we see in the image
        """
        threshold = 200
        val = np.sum(state[:,:,1] > threshold)
        return self._neg_reward if(val == 0) else val 

    def _convert_joint_angle(self, dict_angle):
        arr = np.zeros(7)
        for i,key in enumerate(self._left_joint_names):
            arr[i] = dict_angle[key]
        return arr


    #pid controller
    def _pid_controller(self, action):
        #need to deal with freaking gravity and coriolosis forces on the robot's arm
        #we going to address this by using a pid on other joints to make sure they stay in place

        error_arr = self._start_joint_list - self._joint_angles
        p_arr = np.array([2,1,3,1,2,1,1])
        i_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self._error_collector = self._error_collector + (error_arr/100)
        cmd_arr = error_arr * p_arr + i_arr * self._error_collector
        #cmd_arr = error_arr * p_arr# + i_arr * self._error_collector
        action[np.logical_not(self._active_joint_list)] = cmd_arr[np.logical_not(self._active_joint_list)]
        return action


    def _step(self, action):
        """
        The action will be a [Nx1] velocity input for the arm, where N is the number of active joints
        """
        
        #first is to convert the Nx1 to 7x1
        action_full = np.zeros(self._num_joint)
        action_full[self._active_joint_list] = action #assign them to the full action
        action_full = self._pid_controller(action_full)
        #unpause the simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()

        #set the velocity
        cmd = dict(zip(self._left_joint_names,action_full))
        self._left_arm.set_joint_velocities(cmd)
        #hopefully this will be enough to get one cycle
        self._time_rate.sleep()
        #the state will be the current join angles
        self._joint_angles = self._convert_joint_angle(self._left_arm.joint_angles())
        image = self._last_image
        reward = self._calculate_reward(image)
        #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo()


        #return the current state, reward and bool and something
        return (self._joint_angles, image), reward, False, {"info":"something"}


    def _total_reset(self):
        #unpause gazebo
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()        

        #restart the image
        self._last_image = None

        #initialize interface here. Done here as this is the only point
        #where we are 100% sure we are in an unpaused state
        self._left_arm = baxter_interface.limb.Limb('left')
       # self._left_joint_names = self._left_arm.joint_names()
        self._left_joint_names = ['left_s0','left_s1','left_e0','left_e1','left_w0','left_w1','left_w2']
        self._time_rate = rospy.Rate(10)

        #make sure it's started
        rospy.sleep(1)

        # #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo() 


    def _generate_random_ball_pose(self):
        new_pose = Pose()
        new_pose.orientation.y = 0
        new_pose.position.x = 1.25 + np.random.uniform(-0.1,0.1)
        new_pose.position.y = 0.4 + np.random.uniform(-0.1,0.1)
        new_pose.position.z = 1.1 + np.random.uniform(-0.1,0.1)        

        return new_pose

    def _reset(self):


        #unpause gazebo
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()        

        #restart the image
        self._last_image = None

        #reset the model to the a random position
        joint_positions = self._start_joint_list.copy()
        if(joint_positions is None):
            joint_positions = generate_random_pose()
        else:
            #check if random start
            if(self._random_start):
                random_pose = generate_random_pose()
                joint_positions[self._active_joint_list] = random_pose[self._active_joint_list]

        self._set_model("baxter","robot_description",self._left_joint_names,joint_positions.tolist())        

        ball_pose = self._ball_pose
        if(ball_pose is None):
            ball_pose = self._generate_random_ball_pose()

        #ref = self._ball_prop
        #self._set_link("ball::ball", new_pose, False, ref.mass,ref.ixx,ref.ixy,ref.ixz,ref.iyy,ref.iyz,ref.izz)
        ms = ModelState()
        ms.model_name = "ball"
        ms.pose = ball_pose
        empty_twist = Twist()
        ms.twist = empty_twist
        self._set_link(ms)

        #reset the image
        self._last_image = None

        #wait until we get newest image
        while(self._last_image is None):
            rospy.sleep(0.1)

        # #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo()   

        #step counter
        self._episode_steps = 0;

        #return the newest image as state
        return (joint_positions, self._last_image) 


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            return self._last_image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._last_image)

    def _close(self):
        #close environment
        pass

    def _seed(self, seed=None):
        #seeding function
        pass

