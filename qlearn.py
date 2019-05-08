# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, learning_rate, discount_factor, binsize):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.epsilon = 0.05
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.binsize = binsize

        self.Q = defaultdict(lambda : {0: 0, 1: 0})

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def hash_state(self, state):
        vel = state['monkey']['vel']
        if vel < 0:
            vel = int(max(vel, -3))
        else:
            vel = int(min(vel, 3))

        d = int(state['tree']['dist'] / self.binsize)

        top = int((state['tree']['top'] - state['monkey']['top'])/self.binsize)
        bottom = int((state['monkey']['bot'] - state['tree']['bot'])/self.binsize)

        return tuple([d, top, bottom, vel])

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        feats = self.hash_state(state)

        action = npr.rand() < 0.5

        if self.last_action != None:
            last_feats = self.hash_state(self.last_state)
            action = int(self.Q[feats][1] > self.Q[feats][0])
            max_Q = self.Q[feats][action]
            self.Q[last_feats][self.last_action] += self.learning_rate*(self.last_reward + self.discount_factor * max_Q- self.Q[last_feats][self.last_action])

        self.last_action = action
        self.last_state = state
        return action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)
        # if ii % 20 == 0:
        #     print Counter(hist)


        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

    # trial_vals = [0.1, 0.25, 0.5, 0.75, 0.9, 1]
    #
    # legend_labels = []
    #
    # for l in trial_vals:
    #     for d in trial_vals:
    #         # Select agent.
    agent = Learner(learning_rate=0.25, discount_factor=1, binsize=50)
    # print (l,d)
    # Empty list to save history.
    hist = []
    # Run games.
    num_iters = 200
    run_games(agent, hist, num_iters, 10)

    # calculate running averages
    avgs = [0 for _ in range(num_iters)]
    for i in range(num_iters):
        avgs[i] = np.mean(hist[:i+1])
    print "max:", max(hist)
    plt.plot(avgs)

    plt.xlabel("iteration")
    plt.ylabel("average score")
    # plt.legend(legend_labels)
    plt.show()
