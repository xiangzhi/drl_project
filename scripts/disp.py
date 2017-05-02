
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


def main():
    
    # graph = np.loadtxt(sys.argv[1])
    # title = sys.argv[2]

    #make sure it's the right format
    if(len(sys.argv) != 3):
        print("usage:{} <run_name> <title_name>".format(sys.argv[0]))
        return sys.exit()

    name_performance = "{}.performance".format(sys.argv[1])
    name_generic = "{}.generic".format(sys.argv[1])

    graph_performance = np.loadtxt(name_performance)

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(graph_performance[:,0],graph_performance[:,1])
    ax1.set_ylabel("Reward")
    ax2.plot(graph_performance[:,0],graph_performance[:,2])
    ax2.set_ylabel("Num. Steps")
    # num = np.linspace(0,graph[-1,0],1000)
    # z = np.polyfit(graph[:,0],graph[:,1], 1)
    # z = np.poly1d(z)
    # plt.plot(graph[:,0],graph[:,1])
    # #plt.plot([graph[0,0],graph[-1,0]],z)
    # plt.plot(num,z(num))
    # plt.xlabel("steps")
    # plt.ylabel("avg. reward")
    # # axarr[1].plot(graph[:,0],graph[:,2])
    # # z = np.polyfit(graph[:,0],graph[:,2], 1)
    # # z = np.poly1d(z)
    # # axarr[1].plot(num,z(num))


    # # axarr[1].set_ylabel("avg. length")
    # # axarr[1].set_xlabel("steps")
    # # plt.suptitle(title)
    # plt.title(title)
    plt.show()

    graph_generic = np.loadtxt(name_generic)


    f, (ax1,ax2) = plt.subplots(2, sharex=True)
    #num = np.linspace(0,graph[-1,0],1000)
    #z = np.polyfit(graph[:,0],graph[:,1], 1)
    #z = np.poly1d(z)
    ax1.plot(graph_generic[:,0],graph_generic[:,2])
    ax1.set_ylabel("Reward")
    ax2.plot(graph_generic[:,0],graph_generic[:,3])
    ax2.set_ylabel("Loss")
    #plt.plot([graph[0,0],graph[-1,0]],z)
    # plt.plot(num,z(num))
    # plt.xlabel("steps")
    # plt.ylabel("avg. reward")
    # axarr[1].plot(graph[:,0],graph[:,2])
    # z = np.polyfit(graph[:,0],graph[:,2], 1)
    # z = np.poly1d(z)
    # axarr[1].plot(num,z(num))


    # axarr[1].set_ylabel("avg. length")
    # axarr[1].set_xlabel("steps")
    # plt.suptitle(title)
    #plt.title(title)
    plt.show()



if __name__ == '__main__':
    main()