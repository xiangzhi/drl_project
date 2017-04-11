import gym
import rospy
import time
import numpy as np

import baxter_interface
from std_srvs.srv import Empty
from std_msgs.msg import Bool

from gazebo_msgs.srv import SetModelConfiguration
from lab_baxter_common.camera_toolkit.camera_control_helpers import CameraController
from sensor_msgs.msg import Image
from scipy.misc import imsave


class BaxterEnv(gym.Env):


    def _image_callback(self, image):

        self._last_image = np.fromstring(image.data, dtype=np.uint8).reshape(800,800,3)
        #imsave('out.jpg',self._last_image)


    def __init__(self):

        rospy.init_node('baxter_env_v0',anonymous=True)

        #services used by gazebo 
        self._pause_gazebo = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        self._unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self._reset_baxter = rospy.ServiceProxy('/gazebo/reset_world',Empty)
        self._reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)

        self._set_model = rospy.ServiceProxy('/gazebo/set_model_configuration',SetModelConfiguration)
        rospy.Subscriber('/cameras/left_hand_camera/image',Image,self._image_callback)
        #publisher
        self._enable_pub = rospy.Publisher('/robot/set_super_enable',Bool,queue_size=1)

        self._reset()


        


    def _step(self, action):
        """
        The action will be a [7x1] velocity input for the arm
        """
        
        #unpause the simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()
        print("unpaused")


        #set the velocity
        cmd = dict(zip(self._left_joint_names,action))
        self._left_arm.set_joint_velocities(cmd)
        #hopefully this will be enough to get one cycle
        self._time_rate.sleep()
        #the state will be the current join angles
        state = self._left_arm.joint_angles()

        #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo()

        #return the current state, reward and bool and something
        return state.values(), 1, False, {"info":"something"}

    def _reset(self):

        # #pause the simulation
        # rospy.wait_for_service('/gazebo/pause_physics')
        # self._pause_gazebo()   


        #move the joints to a specific configurations
        self._last_image = None
        #restart the environment
        #self._reset_sim()
        #self._reset_baxter()
        #time.sleep(1)
        #self._enable_pub.publish(True)
        #time.sleep(3)        
        #unpause gazebo
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_gazebo()        

        #the arm we are playing with
        self._left_arm = baxter_interface.limb.Limb('left')
        self._left_joint_names = self._left_arm.joint_names()
        self._time_rate = rospy.Rate(20)
        #return current state
        state = self._left_arm.joint_angles()

        # #pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_gazebo()   

        joint_positions = [-0.272659,1.04701,-0.00123203, 0.49262,-0.0806423,-0.0620532,0.0265941]
        self._set_model("baxter","robot_description",self._left_joint_names,joint_positions)



        return joint_positions


    def _render(self,mode,close):
        #render the enviornment
        pass

    def _close(self):
        #close environment
        pass

    def _seed(self, seed=None):
        #seeding function
        pass

