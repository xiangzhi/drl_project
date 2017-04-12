import unittest

from deeprl_hw2.action_replay_memory_eff import ActionReplayMemoryEff as ActionReplayMemory
from deeprl_hw2.action_replay_memory import ActionReplayMemory as ActionReplayMemoryOld
import numpy as np
import sys
from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor,NumpyPreprocessor



class ActionMemoryTestMethods(unittest.TestCase):

    def test_memory(self):
        memory = ActionReplayMemory(250,4) #test memory
        index = 0
        while(index < 1000):
            axr = np.random.randint(0,100,(84,84,4))
            memory.append(axr,4,5)
            if((index+1)%50 == 0):
                axr = np.random.randint(0,100,(84,84,4))
                memory.end_episode(axr,True)
                index += 1
            index += 1

        for i in range(0,10):
            #some sampling tests
            curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)
            for i,terminal in enumerate(terminal_arr):
                self.assertTrue(np.all(curr_arr[i][:,:,0] == next_arr[i][:,:,1]))
            self.assertTrue(np.sum(reward_arr-5) == 0)
            self.assertTrue(np.sum(action_arr-4) == 0)


    def test_seq(self):
        memory = ActionReplayMemory(250,4) #test memory
        index = 0
        for x in range(0,1000):
            axr = np.random.randint(0,100,(84,84,4))
            memory.append(axr,4,5)

        for i in range(0,10):
            curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)
            for i,terminal in enumerate(terminal_arr):
                empty_arr = np.zeros((84,84))
                for d in range(0,4):
                    self.assertTrue(not np.all(curr_arr[i][:,:,d] == empty_arr))

    def test_detail(self):

        memory = ActionReplayMemory(250,4) #test memory
        memory_old = ActionReplayMemoryOld(250,4)
        index = 0

        h_prep = HistoryPreprocessor(4)
        np_prep = NumpyPreprocessor()
        preprocessors = PreprocessorSequence([h_prep, np_prep])


        for x in range(0,1000):
            axr = np.random.randint(0,100,(84,84))
            prep_state = preprocessors.process_state_for_memory(axr)

            memory.append(prep_state,4,5)
            memory_old.append(prep_state,4,5)

        for t in range(0,10):
            batch_size =32
            indexes = (np.random.randint(0, memory._filled_size, size=batch_size)).tolist()
            curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(batch_size,indexes)
            curr_arr2, next_arr2, reward_arr2, action_arr2, terminal_arr2 = memory_old.sample(batch_size,indexes)
            for i,terminal in enumerate(terminal_arr):
                empty_arr = np.zeros((84,84))
                for d in range(0,4):
                    self.assertTrue(not np.all(curr_arr[i][:,:,d] == empty_arr))    
                    self.assertTrue(np.all(curr_arr[i][:,:,d] == curr_arr2[i][:,:,d])) 

                if(indexes[i] >= 4):
                    self.assertTrue(np.all(curr_arr[i][:,:,1] == memory.survey(indexes[i]-1)))        
                    self.assertTrue(np.all(curr_arr[i][:,:,2] == memory.survey(indexes[i]-2)))        
                    self.assertTrue(np.all(curr_arr[i][:,:,3] == memory.survey(indexes[i]-3)))
                self.assertTrue(np.all(curr_arr[i][:,:,0] == curr_arr2[i][:,:,0])) 


    def test_start_full(self):

        memory = ActionReplayMemory(250,4) #test memory
        memory_old = ActionReplayMemoryOld(250,4)
        index = 0

        h_prep = HistoryPreprocessor(4)
        np_prep = NumpyPreprocessor()
        preprocessors = PreprocessorSequence([h_prep, np_prep])


        for x in range(0,500):
            axr = np.random.randint(0,100,(84,84))
            prep_state = preprocessors.process_state_for_memory(axr)

            memory.append(prep_state,4,5)
            memory_old.append(prep_state,4,5)

        for t in range(0,1):
            batch_size =32
            indexes = [0]
            curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(batch_size,indexes)
            curr_arr2, next_arr2, reward_arr2, action_arr2, terminal_arr2 = memory_old.sample(batch_size,indexes)
            for i,terminal in enumerate(terminal_arr):
                empty_arr = np.zeros((84,84))
                for d in range(0,4):
                    self.assertTrue(not np.all(curr_arr[i][:,:,d] == empty_arr))           
                    self.assertTrue(np.all(curr_arr[i][:,:,d] == curr_arr2[i][:,:,d])) 

                if(indexes[i] > 4):
                    self.assertTrue(np.all(curr_arr[i][:,:,1] == memory.survey(indexes[i]-1)))        
                    self.assertTrue(np.all(curr_arr[i][:,:,2] == memory.survey(indexes[i]-2)))        
                    self.assertTrue(np.all(curr_arr[i][:,:,3] == memory.survey(indexes[i]-3)))
                    self.assertTrue(np.all(curr_arr[i][:,:,0] == curr_arr2[i][:,:,0])) 
                self.assertTrue(np.all(curr_arr[i] == curr_arr2[i]))        

    def test_empty(self):
        memory = ActionReplayMemory(250,4) #test memory
        memory_old = ActionReplayMemoryOld(250,4)        
        h_prep = HistoryPreprocessor(4)
        np_prep = NumpyPreprocessor()
        preprocessors = PreprocessorSequence([h_prep, np_prep])

        for x in range(0,100):
            axr = np.random.randint(0,100,(84,84))
            prep_state = preprocessors.process_state_for_memory(axr)
            memory.append(prep_state,4,5)
            memory_old.append(prep_state,4,5)

        for t in range(0,10):
            batch_size =32
            indexes = (np.random.randint(0,memory._filled_size, size=batch_size)).tolist()
            curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(batch_size,indexes)
            curr_arr2, next_arr2, reward_arr2, action_arr2, terminal_arr2 = memory_old.sample(batch_size,indexes)
            for i,terminal in enumerate(terminal_arr):
                for d in range(0,4):
                    self.assertTrue(np.all(curr_arr[i][:,:,d] == curr_arr2[i][:,:,d])) 

                if(indexes[i] >= 4):
                    self.assertTrue(np.all(curr_arr[i][:,:,1] == memory.survey(indexes[i]-1)))        
                    self.assertTrue(np.all(curr_arr[i][:,:,2] == memory.survey(indexes[i]-2)))        
                    self.assertTrue(np.all(curr_arr[i][:,:,3] == memory.survey(indexes[i]-3)))
                self.assertTrue(np.all(curr_arr[i][:,:,0] == curr_arr2[i][:,:,0])) 


    # def test_memory_deprive(self):
    #     memory = ActionReplayMemory(1000,4)
    #     index = 0 
    #     while(index < 100):
    #         memory.append(index,4,5)
    #         index += 1

    #     for i in range(0,10):
    #         #some sampling tests
    #         curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)
    #         self.assertTrue(np.sum(np.where(curr_arr>101)) == 0) #simple test to see if they are in range

    # def test_end_range(self):
    #     memory = ActionReplayMemory(1000,4)
    #     index = 0 
    #     while(index < 100):
    #         memory.append(index,4,5)
    #         index += 1
    #     max_size = len(memory)
    #     self.assertTrue(memory._memory[max_size-1] != None)     

    # def test_single(self):
    #     memory = ActionReplayMemory(2,4)
    #     index = 0 
    #     last_sample = None
    #     for i in range(0,10):
    #          axr = np.random.randint(0,100,(84,84,4))
    #          last_sample = axr
    #          memory.append(axr,4,5)
    #     self.assertTrue(len(memory) == 2)
    #     axr = np.random.randint(0,100,(84,84,4))
    #     memory.append(axr,4,5)
    #     for x in range(0,10):
    #         curr, next_state, reward, action, terminal = memory.sample(1)
    #         print(curr)
    #         self.assertTrue(np.all(curr[:,:,0] == last_sample))


    # def test_Timing(self):
    #     memory = ActionReplayMemory(100000,4)#test memory
    #     index = 0
    #     while(index  < 100000):
    #         memory.append(index,4,5)
    #         if((index+1)%50 == 0):
    #             memory.end_episode(index+1,True)
    #             index += 1
    #         index += 1      
    #     print('done')   
    #     for i in range(0,10):
    #         #some sampling tests
    #         curr_arr, next_arr, reward_arr, action_arr, terminal_arr = memory.sample(10)

    def test_memory(self):
        memory = ActionReplayMemory(1000000,4)
        # index = 0
        # while(index < 100000):
        #     axr = np.random.randint(0,100,(84,84,4))
        #     memory.append(axr,4,5)
        #     sys.stdout.write('\r{}/{}'.format(index,1000000))
        #     sys.stdout.flush()
        #     index += 1
        print(memory.size())

if __name__ == '__main__':
    #test_memory_size()
    unittest.main()
