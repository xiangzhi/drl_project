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
                 gamma,
                 batch_size,
                 run_name):


      self._sess = sess
      K.set_session(sess)
      self._actor_network = actor_network
      self._critic_network = critic_network
      self._preprocessors = preprocessors
      self._replay_memory = memory
      self._gamma = gamma
      self._batch_size = batch_size


      diff_multiplier = 2

      self._eval_times = 10 #how many episodes we run to evaluate the network
      self._eval_freq = 2000 #how many steps before we do evaluation 
      self._checkin_freq = int(10000 * diff_multiplier)
      self._update_tau = 0.001 
      
      #we make a copy of the preprocessor just for evaluation
      self._preprocessors_eval = self._preprocessors.clone()
      self._run_name =run_name
      self._episodic_recorder = []
      self._performance_recorder = []
   
   
    def compile(self, critic_optimizer,critic_loss_func, actor_learning_rate,critic_input_num,action_dim):

        #compile the Keras model here
        self._critic_network.compile(loss=critic_loss_func, optimizer=critic_optimizer)

        #generate the tf code for running the policy gradient update
        actor_weights = self._actor_network.trainable_weights
        critic_weights = self._critic_network.trainable_weights
        print("start TF compile")
        with tf.name_scope("update_scope"):
          #create an optimizer for the actor 
          opt = tf.train.AdamOptimizer(learning_rate=actor_learning_rate)
          #compute the gradients of the actor
          self._action_tf = tf.placeholder(tf.float32, shape=(None,action_dim))
          
          grad_for_action = tf.gradients(self._critic_network.outputs, self._critic_network.inputs[critic_input_num])[0]
          param_gradient = tf.gradients(self._actor_network.outputs, self._actor_network.trainable_weights, -grad_for_action)
          #apply the gradient
          self._actor_update_opt = opt.apply_gradients(zip(param_gradient, self._actor_network.trainable_weights))

        print("finish TF compile")
        self._sess.run(tf.global_variables_initializer())

        # Log the models to Tensorboard for debugging
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

        #set the backend to test phase
        #K.set_learning_phase(0)
        #run the actor network to get the values
        actions = self._actor_network.predict(state, batch_size=1)
        return actions[0].tolist()

    def cal_q_values(self, states, action_arr):
        """
        Calculate the q values of each state and action using the critic network 
        """
        #make sure the states is a new instance
        states = copy.deepcopy(states)
        #append the action inputs
        states['action_input'] = action_arr
        #get the q_values from network
        q_values = self._target_critic_network.predict(states,batch_size=self._batch_size)
        return q_values

    def select_actions(self, states):
        """
        Calculate the q values of each state and action using the critic network 
        """
        actions = self._target_actor_network.predict(states, batch_size=self._batch_size)
        return actions


    def set_noise_generator(self, noise_generator):
        #the noise generator will be a object with the method generate
        self._noise_generator = noise_generator


    def noise_update(self, action, time_step):
        """
        Given an action, we will add noise to it according to the genrator
        """

        noise = self._noise_generator.generate(action, time_step)
        return action + noise


    def update_step(self, total_step_num):
        """
        Update the neural network by sampling from replay memory
        """

        #K.set_learning_phase(1) #set learning phase
        #first sample from memory
        _curr_state_arr, _next_state_arr, action_list, reward_list, terminal_list  = self._replay_memory.sample(self._batch_size)

        #process all the inputs to make sure they are in the right format
        curr_states = self._preprocessors.process_batch(_curr_state_arr)
        next_states = self._preprocessors.process_batch(_next_state_arr)
        action_arr = np.array(action_list)


        #calculate the Q values for next state
        targeted_actions = self.select_actions(next_states)
        targeted_q_value = self.cal_q_values(next_states,targeted_actions) #batch, size of actions

        #calculate the true Q value at the current state
        y_values = np.zeros(np.shape(targeted_q_value))
        for i in range(len(reward_list)):
          y_values[i] = reward_list[i] #add to all actions's y value
          if(not terminal_list[i]):
            y_values[i] += self._gamma * targeted_q_value[i] #if not terminal, add the q_values for next state

        #make sure critic object is new and append action inputs
        critic_obj = copy.deepcopy(curr_states)
        #add action inputs
        critic_obj['action_input'] = action_arr.copy()
        #train critic network
        training_loss = self._critic_network.train_on_batch(critic_obj, y_values)

        # ------------- update actor network -----------------------------------#
        #get action at current state
        actions = self._actor_network.predict(curr_states,batch_size=self._batch_size)

        #generate feed_dict on the fly based on the states
        feed_dict = {}
        actor_layer_names = curr_states.keys()
        for i,name in enumerate(actor_layer_names):
          feed_dict[self._actor_network.inputs[i]] = curr_states[name].copy()
          feed_dict[self._critic_network.inputs[i]] = curr_states[name].copy()
        #set to learning phase

        #should be 1 if we are also using it in the actor, but right now, its only in the critic
        feed_dict[K.learning_phase()] = 1 # 1 = training phase, 0 = test phase

        #the last one is the critic input
        feed_dict[self._critic_network.inputs[len(actor_layer_names)]] = actions
        #run the TF operation to update it
        self._sess.run(self._actor_update_opt, feed_dict=feed_dict)


        # ------------ update target network ----------------------------------#
        self._target_actor_network = get_soft_target_model_updates(self._target_actor_network, self._actor_network, self._update_tau)
        self._target_critic_network = get_soft_target_model_updates(self._target_critic_network, self._critic_network, self._update_tau)

        #return the training loss of the critic network
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

        #Initialize values
        start_fitting_time = time.time()
        total_step_num = 0
        last_eval_step = 0
        
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

          #process the curernt state for both the network and the memory
          processed_curr_state = self._preprocessors.process_state_for_network(curr_state)
          processed_curr_state_mem = self._preprocessors.process_state_for_memory(curr_state)

          #Loop until the end of the episode or hit the maximum number of episodes
          while(not is_terminal or (max_episode_length != None and step_num >= max_episode_length)):

            #use the actor network select an action
            curr_action = self.select_action(processed_curr_state)

            #add noise to the action
            curr_action = self.noise_update(curr_action,total_step_num)
            #apply the action and save the reward

            #clip the action
            clipped_curr_action = np.clip(curr_action,env.action_space.low,env.action_space.high)

            next_state, reward, is_terminal, debug_info = env.step(clipped_curr_action)
            #env.render()
            #get the current reward
            curr_reward = self._preprocessors.process_reward(reward)
            #process the next frame for both the network and memory
            processed_next_state = self._preprocessors.process_state_for_network(next_state)        
            processed_next_state_mem = self._preprocessors.process_state_for_memory(next_state)        

            #insert into memory
            self._replay_memory.insert(processed_curr_state_mem, processed_next_state_mem, curr_action, curr_reward, is_terminal)

            #update the networks only if the replay memory > batch size
            if(len(self._replay_memory) > self._batch_size):
              training_loss = self.update_step(total_step_num)
              cumulative_loss += training_loss

            #check if we should to a checkpoint save
            if(total_step_num != 0 and (total_step_num%self._checkin_freq == 0)): # or (time.time() - start_fitting_time) > self._total_duration)):
              #do checkin
              self.save_check_point(total_step_num)            

            ##update progress values
            step_num += 1
            total_step_num += 1
            last_eval_step += 1
            cumulative_reward += curr_reward
            #update both rewards and state
            processed_curr_state = processed_next_state
            processed_curr_state_mem = processed_next_state_mem

          #check if we should run an evaluation step and save the rewards
          #This is moved out out of the step loop because it will accidentally restart the environment
          if(last_eval_step > self._eval_freq):
            last_eval_step = 0
            print("\nstart performance evaluation for step:{:09d}".format(total_step_num))
            #evaluate
            avg_reward, avg_length, min_reward, max_reward = self.evaluate(env, self._eval_times, verbose=True, render=False)
            #save the performance
            self._performance_recorder.append((total_step_num, avg_reward, avg_length, min_reward, max_reward))   


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
        return np.mean(reward_arr), np.mean(length_arr), np.min(reward_arr), np.max(reward_arr)

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
            #clip the action
            curr_action = np.clip(curr_action,env.action_space.low,env.action_space.high)

            #curr_action = np.array([0])
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