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

_max_limit = np.array([1,0.75,3,2.6,3,2,3])
_min_limit = np.array([-1.6,-2,-3,0,-3,-1.57,-3])


def generate_random_pose():
    """
    Generates a random baxter pose
    """
    return np.random.uniform(_min_limit, _max_limit).tolist()

class BaxterActionSpace(gym.Space):


    def __init__(self):
        self._max_vel = np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3])
        self._min_vel = np.array([-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3])

    def sample(self):
        """
        Randomly return a sample
        """
        return np.random.uniform(self._min_vel, self._max_vel)

    def contains(self, x):

        #check if the stuff is in range
        return (x < self._max_vel).all() and (x > self._min_vel).all()


class BaxterEnv(gym.Env):


    def _image_callback(self, image):

        self._last_image = np.fromstring(image.data, dtype=np.uint8).reshape(800,800,3)
        #imsave('out.jpg',self._last_image)


    def __init__(self):

        self.action_space = BaxterActionSpace()

        rospy.init_node('baxter_env_v0',anonymous=True)

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

        self._total_reset()
        self._reset()
        

    def _calculate_reward(self, state):
        """
        calculate reward, right now is just how much green pixels we see in the image
        """
        threshold = 200
        val = np.sum(state[:,:,1] > threshold)
        return -1 if(val == 0) else val 

    def _step(self, action):
        """
        The action will be a [7x1] velocity input for the arm
        """
        
        #unpause the simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()

        #set the velocity
        cmd = dict(zip(self._left_joint_names,action))
        self._left_arm.set_joint_velocities(cmd)
        #hopefully this will be enough to get one cycle
        self._time_rate.sleep()
        #the state will be the current join angles
        joint_angles = self._left_arm.joint_angles()
        image = self._last_image
        reward = self._calculate_reward(image)
        #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo()


        #return the current state, reward and bool and something
        return (joint_angles.values(), image), reward, False, {"info":"something"}


    def _total_reset(self):
        #unpause gazebo
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()        

        #restart the image
        self._last_image = None

        #initialize interface here. Done here as this is the only point
        #where we are 100% sure we are in an unpaused state
        self._left_arm = baxter_interface.limb.Limb('left')
        self._left_joint_names = self._left_arm.joint_names()
        self._time_rate = rospy.Rate(10)

        #make sure it's started
        rospy.sleep(1)

        # #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo() 



    def _reset(self):


        #unpause gazebo
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()        

        #restart the image
        self._last_image = None

        #reset the model to the a random position
        joint_positions = generate_random_pose()
        self._set_model("baxter","robot_description",self._left_joint_names,joint_positions)        

        new_pose = Pose()
        empty_twist = Twist()
        new_pose.orientation.y = 0
        new_pose.position.x = 1.25 + np.random.uniform(-0.1,0.1)
        new_pose.position.y = 0.4 + np.random.uniform(-0.1,0.1)
        new_pose.position.z = 1.1 + np.random.uniform(-0.1,0.1)
        #ref = self._ball_prop
        #self._set_link("ball::ball", new_pose, False, ref.mass,ref.ixx,ref.ixy,ref.ixz,ref.iyy,ref.iyz,ref.izz)
        ms = ModelState()
        ms.model_name = "ball"
        ms.pose = new_pose
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


    def _render(self,mode,close):
        #render the enviornment
        pass

    def _close(self):
        #close environment
        pass

    def _seed(self, seed=None):
        #seeding function
        pass

