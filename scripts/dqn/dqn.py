import time
import numpy as np
import copy
import json
import sys

from deeprl_hw2.utils import get_hard_target_model_updates, clone_keras_model
from deeprl_hw2.policy import GreedyEpsilonPolicy

"""Main DQN agent."""

class DQNAgent(object):
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):
        self._network = q_network
        self._preprocessors = preprocessor
        self._replay_memory = memory
        self._policy = policy
        self._gamma = gamma
        self._target_update_freq = target_update_freq
        self._num_burn_in = num_burn_in
        self._train_freq = train_freq
        self._batch_size = batch_size
        
        self._performance_recorder = []
        self._episodic_recorder = []

        self._eval_times = 20#CANNOT SET TO 0
        self._eval_freq = 10000 #how many steps before we do evaluation
        self._checkin_freq = 100000 #how often do we save the weights and do a checkin
        self._skip_frame = 4

        self._keras_custom_layers = None

        #we make a copy of the preprocessor just for evaluation
        self._preprocessors_eval = self._preprocessors.clone()
        self._max_hours = 48
        self._total_duration = 60*60*self._max_hours - 60*1 #10 hours minus the last 2 minutes 

    def add_keras_custom_layers(self, custom):

        self._keras_custom_layers = custom

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

        #compile the Keras model here
        self._network.compile(loss=loss_func, optimizer=optimizer)


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        #use our network to calculate the output
        q_values = self._network.predict(state, batch_size=np.size(state,0))
        return q_values


    def save_models(self):
        file_name = "{}.model".format(self._network.name)
        with open(file_name,"w") as outfile:
          json.dump(self._network.to_json(), outfile)

    def save_check_point(self,step_num):
        #save weights
        file_name = "{}-{:09d}.weights".format(self._network.name, step_num)
        self._network.save_weights(file_name)
        #save performance metric
        performance_csv = np.array(self._performance_recorder)
        np.savetxt("{}.performance".format(self._network.name), performance_csv)     
        #save the episodic informations
        generic_csv = np.array(self._episodic_recorder)
        np.savetxt("{}.generic".format(self._network.name),generic_csv)

        print("\nsaved checkout at step:{:09d}".format(step_num))


    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        #first calculate the q_values
        q_values = self.calc_q_values(np.array([state]))
        #pick an action based on the policy
        actions = self._policy.select_action(q_values=q_values[0,:])
        #return action
        return actions

    def update_policy(self, total_step_num):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        #sample lists of current state, future states and reward, action from the replay memory
        curr_state_arr, next_state_arr, reward_arr, action_arr, terminal_arr  = self._replay_memory.sample(self._batch_size)
        #process all the inputs to make sure they are in the right format
        curr_state_arr = self._preprocessors.process_batch(curr_state_arr)
        next_state_arr = self._preprocessors.process_batch(next_state_arr)
        #create empty array to store q_values
        target_q_values = np.zeros((np.size(reward_arr,0),self._action_size))
        #get the current q_values using the calculate q function
        target_q_values = self.calc_q_values(curr_state_arr)
        #update each target 
        for i, q_values in enumerate(target_q_values):
          target_q_values[i,action_arr[i]] = reward_arr[i]
          if(terminal_arr[i] == 0):
            #not terminal
            #max_q_value = np.argmax(self.calc_q_values(np.array([next_state_arr[i]])))
            max_q_value = np.argmax(self._target_network.predict(np.array([next_state_arr[i]]), batch_size=1))
            target_q_values[i,action_arr[i]] = target_q_values[i,action_arr[i]] + self._gamma * max_q_value

        #train the network
        training_loss = self._network.train_on_batch(curr_state_arr, target_q_values)


        #check whether we want to update the target network
        if(total_step_num % self._target_update_freq == 0):
          self._target_network = get_hard_target_model_updates(self._target_network, self._network)

        return training_loss

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        #copy the current network to the target_network
        self._target_network = clone_keras_model(self._network, self._keras_custom_layers)
        eval_policy = GreedyEpsilonPolicy(0.01)
        self._action_size = env.action_space.n
        start_fitting_time = time.time()
        total_step_num = 0;
        
        #number of iterations
        for num_i in range(0, num_iterations):

          ## Restart Variables and settings
          self._preprocessors.reset() #restart preprocessors
          curr_frame = env.reset() #restart the environment and get the start state
          is_terminal = False
          cumulative_reward = 0
          cumulative_loss = 0
          step_num = 0 #Step number for current episode

          start_time = time.time()

          #for storing over states
          curr_action = 0
          curr_reward = 0
          next_maxed_frame = None

          last_frame = np.zeros(np.shape(curr_frame),dtype=np.uint8)
          mixed_frame = np.maximum(last_frame, curr_frame)
          processed_curr_state = self._preprocessors.process_state_for_memory(mixed_frame)

          #Loop until the end of the episode or hit the maximum number of episodes
          while(not is_terminal or (max_episode_length != None and step_num >= max_episode_length)):


            #use the policy to select the action based on the state
            curr_action = self.select_action(processed_curr_state)

            curr_reward = 0
            #apply the action and save the reward
            #depend on how many frames we skip
            for i in range(0, self._skip_frame):
              last_frame = curr_frame
              curr_frame, reward, is_terminal, debug_info = env.step(curr_action)
              curr_reward += reward
              if(is_terminal):
                break
            #generated the next state
            mixed_frame = np.maximum(last_frame, curr_frame)
            processed_next_state = self._preprocessors.process_state_for_memory(mixed_frame)        

            #insert into memory
            self._replay_memory.insert(processed_curr_state, processed_next_state, curr_action, self._preprocessors.process_reward(curr_reward), is_terminal)

            #update the policy
            training_loss = self.update_policy(total_step_num)
            cumulative_loss += training_loss

            #check if we should run an evaluation step and save the rewards
            if(total_step_num % self._eval_freq == 0):
              print("\nstart performance evaluation for step:{:09d}".format(total_step_num))
              #change policy to use the evaluation policy
              curr_policy = self._policy
              self._policy = eval_policy
              #evaluate
              avg_reward, avg_length = self.evaluate(env, self._eval_times, verbose=True)
              #set back the policy
              self._policy = curr_policy
              #save the performance
              self._performance_recorder.append((total_step_num, avg_reward, avg_length))            

            #check if we should to a checkpoint save
            if(total_step_num%self._checkin_freq == 0 or (time.time() - start_fitting_time) > self._total_duration):
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

        reward_arr, length_arr = self.evaluate_detailed(env,num_episodes,max_episode_length,render,verbose)
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
          last_frame = np.zeros(np.shape(curr_state),dtype=np.uint8)

          while(max_episode_length == None or curr_episode_step <= max_episode_length):

            merge_frames = np.maximum(curr_state,last_frame)
            processed_curr_state = self._preprocessors_eval.process_state_for_network(merge_frames)
            #select action based on the most recent image
            curr_action = self.select_action(processed_curr_state)

            if(render):
              env.render()
            curr_episode_step += 1

            #progress and step through for a fix number of steps according the skip frame number
            for i in range(0,self._skip_frame):
              next_state, reward, is_terminal, info = env.step(curr_action)
              if(is_terminal):
                break
              last_frame = curr_state
              curr_state = next_state
              curr_episode_reward += reward
            if(is_terminal):
              break


          #print("Episode {} ended with length:{} and reward:{}".format(episode_num, curr_episode_step, curr_episode_reward))
          reward_arr[episode_num] = curr_episode_reward
          length_arr[episode_num] = curr_episode_step

          if(verbose):
            sys.stdout.write("\revaluating game: {}/{}".format(episode_num+1, num_episodes))
            sys.stdout.flush()

        return reward_arr, length_arr
