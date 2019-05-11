
import random
import numpy as np 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from util import nearestPoint

EPISODES = 1000


class DQNAgent():
    def __init__(self):
        self.state_size = 2
        self.action_size = 4
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def train(self, gameState, agent_index):
        # env = gym.make('CartPole-v1')
        # agent.load("./save/cartpole-dqn.h5")
        done = False
        batch_size = 32

        for e in range(EPISODES):
            state = gameState.getAgentPosition(agent_index)
            print("\tstate: " + str(state))
            # state = np.reshape(state, [1, self.state_size])
            for time in range(2000):
                # env.render()
                action = self.act(gameState, agent_index, state)
                print("\taction: " + str(action))
                successor = self.getSuccessor(gameState, agent_index, action)
                next_state = successor.getAgentState(agent_index)
                reward = self.getReward(agent_index, successor, gameState)
                # reward
                # done = next_state == successor.getInitialAgentPosition(agent_index)
                # reward = reward if not done else -10
                # next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                            .format(e, EPISODES, time, self.epsilon))
                    break
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
            # if e % 10 == 0:
            #     agent.save("./save/cartpole-dqn.h5")
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, gameState, index, state):
        legal_actions = gameState.getLegalActions(index)
        if np.random.rand() <= self.epsilon:
            print("\trand" + str(self))
            return random.choice(legal_actions)
        act_values = self.model.predict(state)
        print("\tact values: " + str(act_values))
        act_values = act_values[legal_actions]
        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def getSuccessor(self, gameState, index, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(index, action)
        pos = successor.getAgentState(index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def getReward(self, index, new_gs, old_gs):

        reward = 0
        self.getOpponents(old_gs)["position"]
        new_loc = new_gs.getAgentPosition(index)
        old_loc = old_gs.getAgentPosition(index)
        
        # if attacking
        if old_loc[1] > 8:
            # if ate food -> reward: 1
            if new_loc in self.getFood(old_gs):
                reward += 1
            # if ate opponent -> reward: 10
            if new_loc in self.getOpponents(old_gs):
                reward += 20
        # if defending
        else:
            # if opponent ate food -> reward: -1
            if len(self.getOpponentFood(old_gs)):
                reward -= 1
            # if died -> reward: -10
            if new_loc in self.getOpponents(old_gs):
                reward -= 20

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

 
