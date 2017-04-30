import sys
import numpy as np
from scipy import misc


def _calculate_reward(state):
    """
    calculate reward, right now is just how much green pixels we see in the image
    """
    threshold = 200
    val = np.sum(state[:,:,1] > threshold)
    return -1 if(val == 0) else val 

def main():
	image = misc.imread(sys.argv[1])
	reward = _calculate_reward(image)
	if(reward > 0):
		reward = reward/6400.0
	print(reward)


if __name__ == '__main__':
	main()