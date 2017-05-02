import gym
import time
import unittest
import numpy as np
from scipy import misc

#move backwards to the root
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep.preprocessors import PreprocessorSequence, BaxterPreprocessor,HistoryPreprocessor,NumpyPreprocessor


class PreprocessorsTestMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.env =  gym.make("Breakout-v0")
        history_prep = HistoryPreprocessor(4)
        numpy_prep = NumpyPreprocessor()
        bax_prep = BaxterPreprocessor((80,80,3))
        cls.preprocessors = PreprocessorSequence([bax_prep, history_prep,numpy_prep]) #from left to right
        cls.bax_prep = bax_prep
        cls.history_prep = history_prep

    def testBaxterPreprocessor(self):

        for i in range(0,10):
            input_img = np.random.randint(0,255,(800,800,3))
            joint_angle = np.random.randint(0,255,(7,))
            input_state = (joint_angle, input_img)
            output_state = self.bax_prep.process_state_for_network(input_state)
            self.assertTrue(np.shape(output_state[1]) == (80,80,3))
            self.assertTrue(np.all(output_state[0] == joint_angle))

    def testHistoryPreprocessor(self):
        curr_state = np.random.randint(0,255,(84,84))
        self.history_prep.reset()
        last_4_states = np.zeros((4,84,84))
        mem_state = self.history_prep.process_state_for_memory(curr_state)
        last_4_states[0,:,:] = curr_state
        for i in range(1,100):
            nxt_state = np.random.randint(0,255,(84,84))
            #misc.imshow(nxt_state)
            mem_state = self.history_prep.process_state_for_memory(nxt_state)
            last_4_states[i%4,:,:] = nxt_state
            if(i%4 == 0):
                self.assertTrue(np.sum(mem_state - last_4_states) == 0)

        # new_img = np.zeros((84*2,84*2))
        # new_img[0:84,0:84] = mem_state[0,:,:]
        # new_img[0:84,84:84*2] = mem_state[1,:,:]
        # new_img[84:84*2,0:84] = mem_state[2,:,:]
        # new_img[84:84*2,84:84*2] = mem_state[3,:,:]

    def testBaxterSequeunce(self):

        for i in range(0,10):
            input_img = np.random.randint(0,255,(800,800,3))
            joint_angle = np.random.randint(0,255,(7,))
            input_state = (joint_angle, input_img)
            output_state = self.preprocessors.process_state_for_network(input_state)
            jnt_angles = output_state[0]
            img_input = output_state[1]
            #print(output_state[0])
            #print(output_state[1])
            self.assertTrue(np.shape(img_input[:,:,:]) == (80,80,3))
            self.assertTrue(np.all(jnt_angles[:,0] == joint_angle))


    # def testDataSize(self):
    #     curr_state = self.env.reset()
    #     self.preprocessors.reset()
    #     mem_state = self.preprocessors.process_state_for_memory(curr_state)
    #     self.assertTrue(mem_state.dtype == np.uint8)
    #     ori_state = self.preprocessors.process_batch(mem_state)
    #     self.assertTrue(ori_state.dtype == np.float32)

    # def testClone(self):

    #     cloned_prep = self.preprocessors.clone()
    #     #make sure that their preprocessors order is correct
    #     clone_list = cloned_prep.get_preprocessors_name_list()
    #     ori_list = self.preprocessors.get_preprocessors_name_list()
    #     self.assertEqual(len(clone_list), len(ori_list),"size of preprocessor different")
    #     for i, name in enumerate(clone_list):
    #         self.assertEqual(name,ori_list[i],"incorrect order")

    #     #make sure that when restarted both return same things
    #     self.preprocessors.reset()
    #     cloned_prep.reset()
    #     curr_state = self.env.reset()
    #     ori_out = self.preprocessors.process_state_for_network(curr_state)
    #     clone_out = cloned_prep.process_state_for_network(curr_state)
    #     self.assertTrue(np.all(ori_out == clone_out),"output after reset different")

    #     #changed only the original output
    #     nxt_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
    #     ori_out = self.preprocessors.process_state_for_network(nxt_state)
    #     #this test is to make sure that the clone is DEEP
    #     nxt_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
    #     ori_out = self.preprocessors.process_state_for_network(nxt_state)
    #     clone_out = cloned_prep.process_state_for_network(nxt_state)
    #     self.assertTrue(np.any(ori_out != clone_out),"output should be different")

    # def testTiming(self):
    #     #this evaluates how fast the timing of the system
    #     total_time = 0
    #     attempts = 10
    #     num_steps = 100
    #     for i in range(0,attempts):
    #         start_time = time.time()
    #         curr_state = self.env.reset()
    #         for x in range(0,num_steps):
    #             ori_out = self.preprocessors.process_state_for_network(curr_state)
    #             curr_state, reward, terminal, info = self.env.step(np.random.randint(0,self.env.action_space.n))
    #         total_time += (time.time() - start_time)
    #     print("\naverage time:{} per cycle".format(total_time/(attempts*num_steps)))

if __name__ == '__main__':
    unittest.main()