import time
import numpy as np
import copy
import json
import sys

from keras import optimizers


from .utils import get_hard_target_model_updates, clone_keras_model, get_soft_target_model_updates
from .policy import GreedyEpsilonPolicy

import tensorflow as tf
import keras.backend as K
import keras

"""Main Deep  agent."""

class DDPGAgent(object):
    """Class implementing DDPG."""

    def __init__(self,
                 sess, 
                 actor_network,
                 critic_network,
                 preprocessors,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 batch_size,
                 run_name):


      self._sess = sess
      K.set_session(sess)
      self._actor_network = actor_network
      self._critic_network = critic_network
      self._preprocessors = preprocessors
      self._replay_memory = memory
      self._policy = policy
      self._gamma = gamma
      self._target_update_freq = target_update_freq
      self._batch_size = batch_size

      self._eval_times = 10 #how many episodes we run to evaluate the network
      self._eval_freq = 2000 #how many steps before we do evaluation
      self._action_dim = 7  
      self._checkin_freq = 10000
      self._update_tau = 0.001 
      
      #we make a copy of the preprocessor just for evaluation
      self._preprocessors_eval = self._preprocessors.clone()
      self._run_name =run_name
      self._episodic_recorder = []
      self._performance_recorder = []
   
   
    def compile(self, optimizer, loss_func):

        #compile the Keras model here
        self._critic_network.compile(loss=loss_func, optimizer=optimizer)

        #generate the tf code for running the policy gradient update
        actor_weights = self._actor_network.trainable_weights
        critic_weights = self._critic_network.trainable_weights
        print("start TF compile")
        with tf.name_scope("update_scope"):
          #create an optimizer
          opt = tf.train.AdamOptimizer(learning_rate=0.01)
          #opt = optimizers.Adam(lr=0.01)
          #compute the gradients of the ac
          self._action_tf = tf.placeholder(tf.float32, shape=(None,1))
          #action_weights_tf = tf.placeholder()


          #calculate the gradient of the critic network relative to the action selected
          #action_gradient = tf.gradients(self._critic_network.output, self._action_tf)
          grad_for_action = tf.gradients(self._critic_network.outputs, self._critic_network.inputs[1])[0]
          #print(action_gradient)
          # print(actions.shape)
          #print(action_gradient)
          #action_gradient = [(tf.scalar_mul(-1,grad[0]),grad[1]) for grad in action_gradient]
          #action_gradient = tf.scalar_mul(-1, action_gradient)

          #calculate the gradient of the actor network according to the critic weight 
          #action_gradient_tensor = tf.convert_to_tensor(action_gradient, dtype=tf.float32)
          param_gradient = tf.gradients(self._actor_network.outputs, actor_weights, -grad_for_action)
        # #print(param_gradient)
        # #print(actor_weights)



          self._actor_update_opt = opt.apply_gradients(zip(param_gradient, actor_weights))

        print("finish TF compile")
        self._sess.run(tf.global_variables_initializer())


        # #model trainign callbacks
        # self._log_dir = "{}-graph".format(self._run_name)
        # self._actor_ = keras.callbacks.TensorBoard(log_dir=self._log_dir, histogram_freq=0, write_graph=True, write_images=True)



    def save_models(self):
        file_name = "{}.model".format(self._actor_network.name)
        with open(file_name,"w") as outfile:
          json.dump(self._actor_network.to_json(), outfile)

        file_name = "{}.model".format(self._critic_network.name)
        with open(file_name,"w") as outfile:
          json.dump(self._critic_network.to_json(), outfile)

    def save_check_point(self,step_num):
        #save weights
        file_name = "{}-{:09d}.weights".format(self._actor_network.name, step_num)
        self._actor_network.save_weights(file_name)
        file_name = "{}-{:09d}.weights".format(self._critic_network.name, step_num)
        self._critic_network.save_weights(file_name)

        #save performance metric
        performance_csv = np.array(self._performance_recorder)
        np.savetxt("{}.performance".format(self._run_name), performance_csv)     
        #save the episodic informations
        generic_csv = np.array(self._episodic_recorder)
        np.savetxt("{}.generic".format(self._run_name),generic_csv)

        print("\nsaved checkout at step:{:09d}".format(step_num))


    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        Just run the actor network to get the value

        Returns
        --------
        selected action
        """

        # #run the actor network to get the values
        # joint_input = np.array([state[0]]) #this will be a (7,4) list
        # #joint_angles = state[0] #this will be a (7,4) list
        # image_input = np.array([state[1]]) #this will be a (80,80,3,4) array
        # #image_input = state[1] #this will be a (80,80,3,4) array
        # actions = self._actor_network.predict({'image_input':image_input,
        #   'joint_input':joint_input
        #   }, batch_size=1)
        #actions = self._actor_network.predict([[joint_angles],[image_input]],batch_size=1)
        K.set_learning_phase(0)
        actions = self._actor_network.predict({'actor_pendulum_input':np.array([state])}, batch_size=1)

        return actions[0].tolist()

    def cal_q_values(self, states, action_arr):
        """
        Calculate the q values of each state and action using the critic network 
        """
        # joint_angles_list = []
        # image_input_list = []

        # for state in states:
        #   joint_angles_list.append(state[0])
        #   image_input_list.append(state[1])

        # joint_input = np.array(joint_angles_list)
        # image_input = np.array(image_input_list)


        # #get the q_values from the state and action pair from the critic network
        # q_values = self._target_critic_network.predict({'image_input':image_input,
        #   'joint_input':joint_input,
        #   'action_input':action_arr
        #   }, batch_size=self._batch_size)

        # #set to the critic network
        # return q_values

        K.set_learning_phase(0)
        q_values = self._target_critic_network.predict({
          'critic_pendulum_input':np.array(states),
          'action_input':action_arr
          }, batch_size=self._batch_size)
        return q_values

    def select_actions(self, states):
        """
        Calculate the q values of each state and action using the critic network 
        """
        # joint_angles_list = []
        # image_input_list = []

        # for state in states:
        #   joint_angles_list.append(state[0])
        #   image_input_list.append(state[1])

        # joint_input = np.array(joint_angles_list)
        # image_input = np.array(image_input_list)

        # #get the q_values from the state and action pair from the critic network
        # actions = self._target_actor_network.predict({'image_input':image_input,
        #   'joint_input':joint_input
        #   }, batch_size=self._batch_size)

        K.set_learning_phase(0)
        actions = self._target_actor_network.predict({'actor_pendulum_input':np.array(states)}, batch_size=self._batch_size)

        #set to the critic network
        return actions

    def noise_update(self, action, time_step):
        """
        Given an action, we will add noise to it. Uses Ornstein-Uhlenbeck as recommended in the paper 
        """
        means = np.zeros(np.shape(action)) #all means are zero, so no movement around
        thetas = np.ones(np.shape(action)) * 0.15
        sigma = np.ones(np.shape(action)) * 0.2

        #calculate the weiner number
        wiener = np.random.randn(np.size(action)) #we can do this because Wiener process has a gaussian increment

        Noise = thetas * (means - action) + sigma * wiener

        #return action + np.array([(1. / (1. + time_step))])

        return action + Noise 


    def update_step(self, total_step_num):
        """
        Update the neural network by sampling from replay memory
        """

        #K.set_learning_phase(1) #set learning phase
        #first sample from memory
        curr_state_arr, next_state_arr, action_list, reward_list, terminal_list  = self._replay_memory.sample(self._batch_size)
        #process all the inputs to make sure they are in the right format
        curr_state_arr = self._preprocessors.process_batch(curr_state_arr)
        next_state_arr = self._preprocessors.process_batch(next_state_arr)        
        action_arr = np.array(action_list)
        #calculate the q value using the target critic network
        #first get the base q_values from the target critic network
        # curr_q_values = self.cal_q_values(curr_state_arr, action_arr)
        # target_q_values = copy.deepcopy(curr_q_values)

        #print(curr_state_arr)

        targeted_actions = self.select_actions(next_state_arr)
        targeted_q_value = self.cal_q_values(next_state_arr,targeted_actions) #batch, size of actions
        y_values = np.zeros(np.shape(targeted_q_value))

        for i in range(len(reward_list)):
          y_values[i] = reward_list[i] #add to all actions's y value
          if(not terminal_list[i]):
            y_values[i] += self._gamma * targeted_q_value[i] #if not terminal, add the q_values for next state


        #first train the critic
        # joint_list = []
        # image_input_list = []

        # for state in curr_state_arr:
        #   joint_list.append(state[0])
        #   image_input_list.append(state[1])

        # joint_input_x = np.array(joint_list)
        # image_input_x = np.array(image_input_list) 

        #update the critic network given the y value
        # training_loss = self._critic_network.train_on_batch({'image_input':image_input_x,
        #   'joint_input':joint_input_x,
        #   'action_input':action_arr
        #   }, y_values)

        K.set_learning_phase(1)
        state_arr = np.array(curr_state_arr)
        training_loss = self._critic_network.train_on_batch({'critic_pendulum_input':state_arr,
          'action_input':action_arr
          }, y_values)

        #update the actor network using sampled policy gradient
        #the action is predicted
        K.set_learning_phase(0)
        actions = self._actor_network.predict({'actor_pendulum_input':state_arr},batch_size=self._batch_size)
        #update the actor network
        # self._sess.run(self._actor_update_opt, feed_dict={
        #     self._action_tf:actions,
        #     self._actor_network.inputs[0]:image_input_x,
        #     self._actor_network.inputs[1]:joint_input_x
        # })
        K.set_learning_phase(1)
        self._sess.run(self._actor_update_opt, feed_dict={
          self._critic_network.inputs[1]:actions,
          self._actor_network.inputs[0]:state_arr,
          self._critic_network.inputs[0]:state_arr,
        })

        #check whether we want to update the target network
        if(total_step_num % self._target_update_freq == 0):
          self._target_actor_network = get_soft_target_model_updates(self._target_actor_network, self._actor_network, self._update_tau)
          self._target_critic_network = get_soft_target_model_updates(self._target_critic_network, self._critic_network, self._update_tau)
          #print("\nfinish updating target networks")

        #K.set_learning_phase(0) #set learning phase

        return training_loss

    def create_target_networks(self):
        """
        Clones and create target networks
        """
        self._target_critic_network = clone_keras_model(self._critic_network)
        self._target_actor_network = clone_keras_model(self._actor_network)


    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Parameters
        ----------
        env: gym.Env
          This is your environment.
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        #create target network
        self.create_target_networks()

        #save model
        self.save_models()
        self.save_check_point(0)

        #copy the current network to the target_network
        #self._target_network = clone_keras_model(self._network, self._keras_custom_layers)
        #eval_policy = GreedyEpsilonPolicy(0.01)
        #self._action_size = env.action_space.n
        start_fitting_time = time.time()
        total_step_num = 0;
        
        #number of iterations
        for num_i in range(0, num_iterations):

          ## Restart Variables and settings
          self._preprocessors.reset() #restart preprocessors
          curr_state = env.reset() #restart the environment and get the start state
          is_terminal = False
          cumulative_reward = 0
          cumulative_loss = 0
          step_num = 0 #Step number for current episode

          start_time = time.time()

          #for storing over states
          curr_action = 0
          curr_reward = 0

          processed_curr_state = self._preprocessors.process_state_for_memory(curr_state)

          #Loop until the end of the episode or hit the maximum number of episodes
          while(not is_terminal or (max_episode_length != None and step_num >= max_episode_length)):


            #use the policy to select the action based on the state
            #TODO batch this for action
            #processed_curr_state_none_action =
            #print(processed_curr_state) 
            curr_action = self.select_action(processed_curr_state)
            #print(curr_action)
            #add noise to the action
            curr_action = self.noise_update(curr_action,total_step_num)
            curr_reward = 0
            #apply the action and save the reward
            next_state, reward, is_terminal, debug_info = env.step(curr_action)
            env.render()
            #print(curr_action)
            curr_reward = self._preprocessors.process_reward(reward)
            # #depend on how many frames we skip
            # for i in range(0, self._skip_frame):
            #   last_frame = curr_frame
            #   curr_frame, reward, is_terminal, debug_info = env.step(curr_action)
            #   curr_reward += reward
            #   if(is_terminal):
            #     break

            processed_next_state = self._preprocessors.process_state_for_memory(next_state)        

            #insert into memory
            self._replay_memory.insert(processed_curr_state, processed_next_state, curr_action, curr_reward, is_terminal)


            #update the policy
            training_loss = self.update_step(total_step_num)
            cumulative_loss += training_loss

            #check if we should run an evaluation step and save the rewards
            if(total_step_num != 0 and total_step_num % self._eval_freq == 0):
              print("\nstart performance evaluation for step:{:09d}".format(total_step_num))
              #change policy to use the evaluation policy
              #curr_policy = self._policy
              #self._policy = eval_policy
              #evaluate
              avg_reward, avg_length = self.evaluate(env, self._eval_times, verbose=True, render=False)
              #set back the policy
              #self._policy = curr_policy
              #save the performance
              self._performance_recorder.append((total_step_num, avg_reward, avg_length))            

            #check if we should to a checkpoint save
            if(total_step_num != 0 and (total_step_num%self._checkin_freq == 0)): # or (time.time() - start_fitting_time) > self._total_duration)):
              #do checkin
              self.save_check_point(total_step_num)            

            ##update progress values
            step_num += 1
            total_step_num += 1
            cumulative_reward += curr_reward
            processed_curr_state = processed_next_state


          #for tracking purposes
          sys.stdout.write("\r{:09d} ep:{:04d}, len:{:04d}, reward:{:.4f}, loss:{:.5f}, time_per_step:{:.5f}".format(total_step_num, num_i, step_num, cumulative_reward, cumulative_loss/step_num, (time.time() - start_time)/step_num))
          sys.stdout.flush()
          #save these generic informations, good for debugging?
          self._episodic_recorder.append((num_i,step_num,cumulative_reward,cumulative_loss))


    def evaluate(self, env, num_episodes,max_episode_length=None, render=False, verbose=False):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print("starting evaluation")
        reward_arr, length_arr = self.evaluate_detailed(env,num_episodes,max_episode_length,render,verbose)
        if(verbose):
          print("\nfinish evaulate, average reward:{}".format(np.mean(reward_arr)))
        return np.mean(reward_arr), np.mean(length_arr)

    def evaluate_detailed(self, env, num_episodes,max_episode_length=None, render=False, verbose=False):
        """
        An evaluation of the network with more detail information
        """

        reward_arr = np.zeros((num_episodes))
        length_arr = np.zeros((num_episodes))
        frame_num = 0


        #number of episodes
        for episode_num in range(0, num_episodes):

          self._preprocessors_eval.reset()
          curr_episode_reward = 0
          curr_episode_step = 0
          #get the initial state
          curr_state = env.reset()
          if(render):
            env.render()

          frame_num = 0
          curr_reward = 0
          curr_action = 0
          is_terminal = False

          while(max_episode_length == None or curr_episode_step <= max_episode_length):
            processed_curr_state = self._preprocessors_eval.process_state_for_network(curr_state)
            #select action based on the most recent image
            curr_action = self.select_action(processed_curr_state)
            if(render):
              env.render()
            curr_episode_step += 1
            #progress and step through for a fix number of steps according the skip frame number
            next_state, reward, is_terminal, info = env.step(curr_action)
            if(is_terminal):
              break
            curr_state = next_state
            curr_episode_reward += self._preprocessors_eval.process_reward(reward)


          #print("Episode {} ended with length:{} and reward:{}".format(episode_num, curr_episode_step, curr_episode_reward))
          reward_arr[episode_num] = curr_episode_reward
          length_arr[episode_num] = curr_episode_step

          if(verbose):
            sys.stdout.write("\revaluating game: {}/{} length:{} and reward:{}".format(episode_num+1, num_episodes, curr_episode_step, curr_episode_reward))
            sys.stdout.flush()
        return reward_arr, length_arr
