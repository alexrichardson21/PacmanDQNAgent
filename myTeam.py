# myTeam.py
# ---------
# Licensing Infoesmation:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# TO DISCUSS:
# Walkthru
# Replay Func
# Agent state vs position
#   Normalizing state values
# Actions vs. Legal Actions
# Reward Func

import random
import time
import math
import json
import os
from util import nearestPoint
from collections import deque

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from game import Directions
from captureAgents import CaptureAgent

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffDQNAgent', second='DefDQNAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class DQNAgent(CaptureAgent):
    def registerInitialState(self, gs):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''

        '''
        Your initialization code goes here, if you need any.
        '''
        print("REGISTERING INITIAL STATE... \n\n")

        train = True
        
        self.EPISODES = 10000
        self.memory = deque(maxlen=2000)
        self.alpha = 0.05
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.002
        self.epsilon = self.epsilon_min

        self.start = gs.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gs)
        self.actions = ['Stop', 'North', 'South', 'East', 'West']

        cols = len(gs.data.layout.layoutText[0])
        rows = len(gs.data.layout.layoutText)

        self.input_shape = rows*cols
        self.output_shape = len(self.actions)
        
        if os.path.exists('DQNAgent%d.h5' % self.index):
            self.model.load_weights("agent%d.h5" % self.index)
        else:
            self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_shape))
        model.add(Dense(32))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    # DEPRECATED
    def train(self, gs):

        batch_size = 32

        print("Beginning training ...")
        for e in range(self.EPISODES):
            state = gs.getAgentState(self.index)
            legal_actions = gs.getLegalActions(self.index)

            best_index = self.act(gs)
            best_action = self.chooseAction(gs)

            next_gs = self.getSuccessor(gs, best_action)
            next_state = next_gs.getAgentState(self.index)
            reward = self.getReward(next_gs, gs)

            self.remember(gs, best_index, reward, next_gs)

            with open("memory.json", "w") as write_file:
                json.dump((self.index,
                           gs.getAgentPosition(self.index),
                           best_action, reward,
                           next_gs.getAgentPosition(self.index)
                           ), write_file)

            gs = next_gs

            if len(self.memory) > batch_size:
                self.replay(batch_size)

            if (e % 100 == 0):
                print("Episode: %d" % e)

        self.model.save_weights("agent%d.h5" % self.index)
        print('Finished Training!')

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        # Samples random memories of batch_size
        minibatch = random.sample(self.memory, batch_size)

        # For each memory
        avg_loss = []
        for gs, action, reward, next_gs in minibatch:
            state = gs.getAgentState(self.index)
            next_state = next_gs.getAgentState(self.index)

            # Update to q value
            gs_q_vals = self.model.predict(self.preprocessGS(gs))
            best_q_val = np.amax(gs_q_vals[0])
            next_best_q_val = np.amax(
                self.model.predict(self.preprocessGS(next_gs))[0])

            diff = (reward + self.gamma * next_best_q_val) - best_q_val
            gs_q_vals[0][self.actions.index(action)] = diff

            loss = self.model.fit(self.preprocessGS(gs),
                                  gs_q_vals, epochs=1, verbose=0)

            avg_loss += loss.history['loss']

        # print("Replay Avg Loss: " + str(np.average(avg_loss)))

        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def getSuccessor(self, gs, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gs.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gs):
        """
        Picks among actions randomly.
        """
        # state = gs.getAgentPosition(self.index)
        # actions = gs.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''
        "*** YOUR CODE HERE ***"
        batch_size = 16

        # Update memory if possible
        last_gs = self.getPreviousObservation()
        if last_gs:
            next_gs = self.getCurrentObservation()
            if next_gs.data.timeleft <= 5:
                self.model.save('DQNAgent%d.h5' % self.index)
            reward = self.getReward(gs, last_gs)
            action = self.getDirection(last_gs.getAgentPosition(
                self.index), gs.getAgentPosition(self.index))
            self.memory.append((last_gs, action, reward, gs))

            with open("memory.json", "w") as write_file:
                json.dump((self.index,
                        last_gs.getAgentPosition(self.index),
                        action, reward,
                        gs.getAgentPosition(self.index)
                        ), write_file)

        # Replay
        if len(self.memory) > batch_size:
            self.replay(batch_size)
        
        legal_actions = gs.getLegalActions(self.index)

        # Random Action
        if np.random.rand() <= self.epsilon:
            best_action = random.choice(legal_actions)
        # Best Action
        else:
            act_values = self.model.predict(self.preprocessGS(gs))
            legal_actions_i = [self.actions.index(a) for a in legal_actions]
            best_action = np.argmax(act_values[0][legal_actions_i])
            best_action = self.actions[legal_actions_i[best_action]]

        return best_action  # returns action

    def preprocessGS(self, gs):
        data = []
        layout = gs.data.layout.layoutText
        # new_layout = np.zeros(((16,)))
        for i, row in enumerate(layout):
            new_row = row.replace(" ", "0") \
                .replace("%", "5") \
                .replace(".", "6") \
                .replace("o", "7")
            data += [float(x) / float(7) for x in list(new_row)]
        # + [str(self.actions.index(action))]
        return np.reshape(np.asarray(data, dtype=float).flatten(), (1, self.input_shape))

    def min_dist_to_food(self, gs, agent_pos):
        food_pos = []
        for i, r in enumerate(self.getFood(gs).data):
            for j, c in enumerate(r):
                if self.getFood(gs).data[i][j]:
                    food_pos += [(i, j)]

        return np.min([self.getMazeDistance(agent_pos, f)
                       for f in food_pos])

    def min_dist_to_op(self, gs, agent_pos):
        op_pos = [gs.getAgentPosition(i)
                  for i in self.getOpponents(gs)]

        return np.min([self.getMazeDistance(agent_pos, f)
                       for f in op_pos])

    def isAgentDead(self, new_gs, old_gs):
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        op_pos = [new_gs.getAgentPosition(i)
                  for i in self.getOpponents(new_gs)]

        if old_loc in op_pos and new_loc == self.start:
            return True
        return False
    
    def isOpDead(self, new_gs, old_gs):
        op_i = self.getOpponents(new_gs)
        new_op_locs = [new_gs.getAgentPosition(i) for i in op_i]
        old_op_locs = [old_gs.getAgentPosition(i) for i in op_i]
        old_loc = old_gs.getAgentPosition(self.index)
        new.gs.getAgentState.start

        if old_loc in op_pos and new_loc == self.start:
            return True
        return False

    def getDirection(self, prev_pos, curr_pos):
        if prev_pos[0] < curr_pos[0]:
            return 'West'
        elif prev_pos[0] > curr_pos[0]:
            return 'East'
        else:
            if prev_pos[1] < curr_pos[1]:
                return 'North'
            elif prev_pos[1] > curr_pos[1]:
                return 'South'
            else:
                return 'Stop'

class OffDQNAgent(DQNAgent):
    def getReward(self, new_gs, old_gs):
        # init
        new_agent = new_gs.getAgentState(self.index)
        old_agent = old_gs.getAgentState(self.index)
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        # op_pos = [new_gs.getAgentPosition(i)
        #           for i in self.getOpponents(new_gs)]
        food_pos = []
        for i, r in enumerate(self.getFood(old_gs).data):
            for j, c in enumerate(r):
                if self.getFood(old_gs).data[i][j]:
                    food_pos += [(i, j)]
        
        reward = 0
        
        # Move closer to food
        reward += 20.0 * (self.min_dist_to_food(old_gs, old_loc) -
                                self.min_dist_to_food(old_gs, new_loc)) / float(old_agent.numCarrying + 1) - 3.0
        
        # No movement
        if old_loc == new_loc:
            reward -= 4.0

        # Close to Food
        reward += (50.0 - self.min_dist_to_food(old_gs, new_loc)) / 10.0

        # Holding too many
        reward -= new_agent.numCarrying * 1.5
        
        # pick up dot
        r, c = new_loc
        if self.getFood(old_gs).data[r][c]:
            reward += 50.0
        
        # return dots to side
        reward += 200.0 * (new_agent.numReturned - old_agent.numReturned)

        # died 
        if self.isAgentDead(new_gs, old_gs):
            reward -= 500.0
        
        # close to op
        if new_agent.isPacman:
            old_distances = min(
                old_gs.agentDistances[self.getOpponents(old_gs)])
            new_distances = min(
                old_gs.agentDistances[self.getOpponents(old_gs)])
            if new_distances < 4:
                reward -= (5 - new_distances) * 20.0

        with open("off_rewards.json", "w") as write_file:
            json.dump(reward, write_file)

        return reward


class DefDQNAgent(DQNAgent):
    def getReward(self, new_gs, old_gs):
        # init
        new_agent = new_gs.getAgentState(self.index)
        old_agent = old_gs.getAgentState(self.index)
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        # op_pos = [old_gs.getAgentPosition(i)
        #           for i in self.getOpponents(old_gs)]
        op_indices = self.getOpponents(old_gs)

        reward = 0
        

        # if not (new_agent.isPacman):
        #     min_dist_to_op = self.min_dist_to_op(old_gs, new_loc)
        #     reward = float(50) - min_dist_to_op
            
        #     if(min_dist_to_op == 0):
        #         reward += 200
        
        if new_agent.isPacman:
            reward -= 50


        # living penalty while on defensive side -> reward = -.03
        # if not (new_agent.isPacman):
        #     reward -= .03

        #     # capture opponent -> 20
        #     min_dist_to_op = self.min_dist_to_op(old_gs, new_loc)
        #     if(min_dist_to_op == 0):
        #         reward += 20

        #     # Opponent far -> -1 Opponent close -> 1
        #     else:
        #         reward += math.abs(min_dist_to_op / float(50) - 1)

        # living penalty while on offensive side -> reward = -.05
        # else:
        #     reward -= .05

        # # died -> -50
        # if self.isDead(new_gs, old_gs):
        #     reward -= 50

        # # if opponent returns dots -> reward = -3 * num returned
        # old_num_returned = [old_gs.getAgentState(i).numReturned
        #                     for i in op_indices]
        # new_num_returned = [new_gs.getAgentState(i).numReturned
        #                     for i in op_indices]
        # reward -= 3 * (sum(new_num_returned) - sum(old_num_returned))

        with open("def_rewards.json", "w") as write_file:
            json.dump(reward, write_file)

        return reward
