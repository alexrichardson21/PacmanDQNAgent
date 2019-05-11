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
from util import nearestPoint
from collections import deque

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from game import Directions
# import game
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
        self.EPISODES = 1000
        # self.index = index
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.start = gs.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gs)
        self.actions = ['Stop', 'North', 'South', 'East', 'West']

        cols = len(gs.data.layout.layoutText[0])
        rows = len(gs.data.layout.layoutText)

        self.input_shape = rows*cols
        self.output_shape = len(self.actions)
        self.model = self._build_model()
        self.train(gs)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.input_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def train(self, gs):

        batch_size = 32

        print("Beginning training ...")
        for e in range(self.EPISODES):
            state = gs.getAgentState(self.index)
            legal_actions = gs.getLegalActions(self.index)

            best_index = self.act(gs)
            best_action = self.actions[best_index]

            next_gs = self.getSuccessor(gs, best_action)
            next_state = next_gs.getAgentState(self.index)
            reward = self.getReward(next_gs, gs)

            self.remember(gs, best_index, reward, next_gs)

            gs = next_gs

            if len(self.memory) > batch_size:
                self.replay(batch_size)

            if (e % 100 == 0):
                print("Episode: %d" % e)

            # scores = [long("-inf")] * len(self.actions)
            # for i, a in enumerate(self.actions):
            #     if a in legal_actions:
            #         data = self.preprocessGS(gs, a)
            #         scores[i] = self.model.evaluate(data)
            # best_action = self.actions[np.argmax(scores)]

            # successor = self.getSuccessor(gs, best_action)
            # reward = self.getReward(successor, gs)

            # self.model.fit(np.argmax(scores), reward)

            # action = self.act(gs, state)
            # successor = self.getSuccessor(gs, action)

            # next_state = successor.getAgentState(self.index)
            # reward = self.getReward(successor, gs)
            # self.remember(state, action, reward, next_state.start.pos, gs)

            # gs = successor

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, gs):
        legal_actions = gs.getLegalActions(self.index)

        # Random Action
        if np.random.rand() <= self.epsilon:
            return self.actions.index(random.choice(legal_actions))

        # Predict best action
        act_values = self.model.predict(self.preprocessGS(gs))
        # Set illegal actions to 0
        # for i, a in enumerate(self.actions):
        #     if a not in gs.getLegalActions(self.index):
        #         act_values[0][i] = 0
        # best_action = np.argmax(act_values[0])
        legal_actions = []
        for i, a in enumerate(self.actions):
            if a not in gs.getLegalActions(self.index):
                legal_actions += [i]
        best_action = np.argmax(legal_actions)

        # Trains model for illegal actions
        self.model.fit(self.preprocessGS(gs), act_values, epochs=1, verbose=0)

        return best_action # returns action

    def replay(self, batch_size):
        # Samples random memories of batch_size
        minibatch = random.sample(self.memory, batch_size)

        # For each memory
        avg_loss = []
        for gs, action, reward, next_gs in minibatch:
            state = gs.getAgentState(self.index)
            next_state = next_gs.getAgentState(self.index)
            # legal_actions = gs.getLegalActions(self.index)

            # Update to q value
            target = reward + self.gamma * \
                np.amax(self.model.predict(self.preprocessGS(next_gs))[0])

            target_f = self.model.predict(self.preprocessGS(gs))
            # Sets illegal actions to 0
            # for i, a in enumerate(self.actions):
            #     if a not in gs.getLegalActions(self.index):
            #         target_f[0][i] = 0
            # Sets action q value to target
            target_f[0][action] = target
            
            loss = self.model.fit(self.preprocessGS(gs), target_f, epochs=1, verbose=0)
            avg_loss += loss.history['loss']
        print("Replay Avg Loss: " + str(np.average(avg_loss)))
        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





        #     # Predicts actions values for next_state
        #     act_vals = self.model.predict(np.reshape(
        #         next_state, (1, self.state_size)))[0]
        #     possible_act_vals = act_vals[[
        #         self.actions.index(a) for a in legal_actions]]

        #     # Updates target values for new action value: target
        #     target = (reward + self.gamma * np.amax(possible_act_vals))
        #     target_f = self.model.predict(
        #         np.reshape(state, (1, self.state_size)))
        #     target_f[0][self.actions.index(action)] = target

        #     # Trains model on state to target_f
        #     self.model.fit(np.reshape(state, (1, self.state_size)),
        #                    target_f, epochs=1, verbose=0)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

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

    def getReward(self, new_gs, old_gs):

        reward = 0
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        new_agent = new_gs.getAgentState(self.index)
        old_agent = old_gs.getAgentState(self.index)
        op_pos = [old_gs.getAgentPosition(i)
                  for i in self.getOpponents(old_gs)]

        # first few moves
        if old_gs.data.timeleft > 1150:
            reward = -.1 * (np.average([self.getMazeDistance(old_loc, op) for op in op_pos]))
        
        # if attacking
        elif old_agent.isPacman:
            # if ate food -> reward: 1
            r, c = new_loc
            if self.getFood(old_gs).data[r][c]:
                reward += 1
            # if ate opponent -> reward: 10
            if (old_loc in op_pos) and (new_loc == self.start):
                reward -= 20

        # if defending
        else:
            # if died -> reward: -10
            op_pos = [old_gs.getAgentPosition(i) 
                      for i in self.getOpponents(old_gs)]
            if new_loc in op_pos:
                reward += 20

        return reward

    def chooseAction(self, gs):
        """
        Picks among actions randomly.
        """
        state = gs.getAgentPosition(self.index)
        actions = gs.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''
        "*** YOUR CODE HERE ***"
        return self.actions[self.act(gs)]
        # Random choice
        # if np.random.rand() <= self.epsilon:
        #     return random.choice(actions)

        # # Q scores approximated by model
        # scores = [long("-inf")] * len(self.actions)
        # for i, a in enumerate(self.actions):
        #     if a in actions:
        #         data = self.preprocessGS(gs, a)
        #         scores[i] = self.model.evaluate(data)
        # return self.actions[np.argmax(scores)]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def preprocessGS(self, gs):
        data = []
        layout = gs.data.layout.layoutText
        for i, row in enumerate(layout):
            row = row.replace(" ", "0") \
                .replace("%", "5") \
                .replace(".", "6") \
                .replace("o", "7")
            data += [int(x) / 7 for x in list(row)]
        # + [str(self.actions.index(action))]
        return np.reshape(np.asarray(data, dtype=int).flatten(), (1, self.input_shape))
    
    def starting_reward(self, new_gs, old_gs):
        new_loc = new_gs.getAgentPosition(self.index)
        op_pos = [new_gs.getAgentPosition(i)
                  for i in self.getOpponents(new_gs)]
        return (- np.amin([self.getMazeDistance(new_loc, op) 
                        for op in op_pos]) / float(50)) + 1
    
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
    
    def isDead(self, new_gs, old_gs):
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        op_pos = [new_gs.getAgentPosition(i)
                  for i in self.getOpponents(new_gs)]
        
        if old_loc in op_pos and new_loc == self.start:
            return True
        return False
        
class OffDQNAgent(DQNAgent):
    def getReward(self, new_gs, old_gs):
        # init
        new_agent = new_gs.getAgentState(self.index)
        old_agent = old_gs.getAgentState(self.index)
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        op_pos = [new_gs.getAgentPosition(i)
                  for i in self.getOpponents(new_gs)]
        food_pos = []
        for i, r in enumerate(self.getFood(old_gs).data):
            for j, c in enumerate(r):
                if self.getFood(old_gs).data[i][j]:
                    food_pos += [(i,j)] 
        
        reward = 0
        
        # living penalty while on offensive side -> -.03
        if old_agent.isPacman and new_agent.isPacman:
            reward -= .03
        
        # living penalty while on defensive side -> -.05
        if not old_agent.isPacman and not new_agent.isPacman:
            reward -= .05

        # move closer to op in first 100 moves -> reward ^ if closer
        # if old_gs.data.timeleft > 1100:
        #     reward += self.starting_reward(new_gs, old_gs)

        # pick up dot -> 2
        r, c = new_loc
        if self.getFood(old_gs).data[r][c]:
            reward += 2
        
        # food far -> -1    food close -> 1
        else:
            closest_food = np.min([self.getMazeDistance(new_loc, f)
                                   for f in food_pos])
            reward -= self.min_dist_to_food(new_gs, new_loc) / float(50) * 2 - 1
       
        
        # return dots to side -> reward = 3 * num carrying
        if old_agent.isPacman and not new_agent.isPacman:
            reward += 3 * old_agent.num_carrying
        
        # died -> reward = -20
        if old_loc in op_pos and new_loc == self.start:
            reward -= 20
        
        print(reward)
        return reward

class DefDQNAgent(DQNAgent):
    def getReward(self, new_gs, old_gs):
        # init
        new_agent = new_gs.getAgentState(self.index)
        old_agent = old_gs.getAgentState(self.index)
        new_loc = new_gs.getAgentPosition(self.index)
        old_loc = old_gs.getAgentPosition(self.index)
        op_pos = [old_gs.getAgentPosition(i)
                  for i in self.getOpponents(old_gs)]
        op_indices = self.getOpponents(old_gs)
        
        reward = 0
        
        # living penalty while on defensive side -> reward = -.03
        if not old_agent.isPacman and not new_agent.isPacman:
            reward -= .03

        # capture opponent -> 20
        min_dist_to_op = self.min_dist_to_op(old_gs, new_loc)

        if (min_dist_to_op == 0) and not (new_agent.isPacman):
            reward += 20
        
        # Opponent far -> 0 Opponent close -> 1
        else:
            reward += math.abs(min_dist_to_op / float(100) - 1)

        # died -> -50
        if self.isDead(new_gs, old_gs):
            reward -= 50
    
        # if opponent returns dots -> reward = -2 * num returned
        old_num_returned = [old_gs.getAgentState(i).numReturned
                             for i in op_indices]
        new_num_returned = [new_gs.getAgentState(i).numReturned 
                             for i in op_indices]
        reward -= 2 * (sum(new_num_returned) - sum(old_num_returned))
        
        return reward
