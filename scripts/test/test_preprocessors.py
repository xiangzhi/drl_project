import gym
import time
import unittest
import numpy as np
from scipy import misc
from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor,NumpyPreprocessor


class PreprocessorsTestMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.env =  gym.make("Breakout-v0")
        history_prep = HistoryPreprocessor(4)
        atari_prep = AtariPreprocessor((84,84),0,999)
        numpy_prep = NumpyPreprocessor()
        cls.preprocessors = PreprocessorSequence([atari_prep, history_prep, numpy_prep]) #from left to right
        cls.atari_prep = atari_prep

    def testReward(self):
        #if Normalized
        # self.assertEqual(self.preprocessors.process_reward(0),-1)
        # self.assertEqual(self.preprocessors.process_reward(-1),-1)
        # self.assertEqual(self.preprocessors.process_reward(999),1)
        # self.assertEqual(self.preprocessors.process_reward(1000),1)
        # self.assertEqual(self.preprocessors.process_reward(1100),1)
        # self.assertTrue(self.preprocessors.process_reward(999) > 0.97)
        # self.assertTrue(self.preprocessors.process_reward(500) < 0.01 and self.preprocessors.process_reward(500) > -0.01)
        # self.assertTrue(self.preprocessors.process_reward(1) > -1)
        self.assertEqual(self.preprocessors.process_reward(1),1)
        self.assertEqual(self.preprocessors.process_reward(100),1)
        self.assertEqual(self.preprocessors.process_reward(400),1)
        self.assertEqual(self.preprocessors.process_reward(0),0)
        self.assertEqual(self.preprocessors.process_reward(-1),0)
        self.assertEqual(self.preprocessors.process_reward(-100),0)


    def testSizing(self):
        curr_state = self.env.reset()
        last_4_states = np.zeros((84,84,4))
        self.preprocessors.reset()
        mem_state = self.preprocessors.process_state_for_memory(curr_state)
        for i in range(1,100):
            nxt_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
            #misc.imshow(nxt_state)
            mem_state = self.preprocessors.process_state_for_memory(nxt_state)
            self.assertTrue(np.shape(mem_state), (84,84,4)) #size always correct
            #see whether the history was correct
            if(i >= 2):
                self.assertTrue(np.sum(mem_state[:,:,:]) != 0)
            else:
                self.assertTrue(np.sum(mem_state[:,:,3]) == 0)

    def testHistory(self):
        curr_state = self.env.reset()
        self.preprocessors.reset()
        last_4_states = np.zeros((84,84,4))
        mem_state = self.preprocessors.process_state_for_memory(curr_state)
        last_4_states[:,:,0] = self.atari_prep.process_state_for_network(curr_state)
        for i in range(1,100):
            nxt_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
            #misc.imshow(nxt_state)
            mem_state = self.preprocessors.process_state_for_memory(nxt_state)
            last_4_states[:,:,i%4] = self.atari_prep.process_state_for_network(nxt_state)
            if(i%4 == 0):
                self.assertTrue(np.sum(mem_state - last_4_states) == 0)

        new_img = np.zeros((84*2,84*2))
        new_img[0:84,0:84] = mem_state[:,:,0]
        new_img[0:84,84:84*2] = mem_state[:,:,1]
        new_img[84:84*2,0:84] = mem_state[:,:,2]
        new_img[84:84*2,84:84*2] = mem_state[:,:,3]
        misc.imshow(new_img)

    def testDataSize(self):
        curr_state = self.env.reset()
        self.preprocessors.reset()
        mem_state = self.preprocessors.process_state_for_memory(curr_state)
        self.assertTrue(mem_state.dtype == np.uint8)
        ori_state = self.preprocessors.process_batch(mem_state)
        self.assertTrue(ori_state.dtype == np.float32)

    def testClone(self):

        cloned_prep = self.preprocessors.clone()
        #make sure that their preprocessors order is correct
        clone_list = cloned_prep.get_preprocessors_name_list()
        ori_list = self.preprocessors.get_preprocessors_name_list()
        self.assertEqual(len(clone_list), len(ori_list),"size of preprocessor different")
        for i, name in enumerate(clone_list):
            self.assertEqual(name,ori_list[i],"incorrect order")

        #make sure that when restarted both return same things
        self.preprocessors.reset()
        cloned_prep.reset()
        curr_state = self.env.reset()
        ori_out = self.preprocessors.process_state_for_network(curr_state)
        clone_out = cloned_prep.process_state_for_network(curr_state)
        self.assertTrue(np.all(ori_out == clone_out),"output after reset different")

        #changed only the original output
        nxt_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
        ori_out = self.preprocessors.process_state_for_network(nxt_state)
        #this test is to make sure that the clone is DEEP
        nxt_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
        ori_out = self.preprocessors.process_state_for_network(nxt_state)
        clone_out = cloned_prep.process_state_for_network(nxt_state)
        self.assertTrue(np.any(ori_out != clone_out),"output should be different")

    def testTiming(self):
        #this evaluates how fast the timing of the system
        total_time = 0
        attempts = 10
        num_steps = 100
        for i in range(0,attempts):
            start_time = time.time()
            curr_state = self.env.reset()
            for x in range(0,num_steps):
                ori_out = self.preprocessors.process_state_for_network(curr_state)
                curr_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
            total_time += (time.time() - start_time)
        print("\naverage time:{} per cycle".format(total_time/(attempts*num_steps)))

if __name__ == '__main__':
    unittest.main()