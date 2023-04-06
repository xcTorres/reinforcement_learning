import math
import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.0001

MEMORY_SIZE = 10000
BATCH_SIZE = 128

EXPLORATION_MAX = 0.9
EXPLORATION_MIN = 0.005
EXPLORATION_DECAY = 1000

# How often to update the target network
UPDATE_TARGET_FREQUENCY = 10
optimizer=Adam(lr=LEARNING_RATE)

class DDQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = self.create_q_network()
        self.model_target = self.create_q_network()

    
    def create_q_network(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.observation_space,), activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.memory)), size=BATCH_SIZE)

        # Using list comprehension to sample from replay buffer
        state_sample  = np.array([self.memory[i][0] for i in indices])
        action_sample = np.array([self.memory[i][1] for i in indices])
        rewards_sample = np.array([self.memory[i][2] for i in indices])
        state_next_sample  = np.array([self.memory[i][3] for i in indices])
        done_sample = np.array([self.memory[i][4] for i in indices])


        current_q_batch = self.model.predict(state_next_sample).squeeze()
        max_action_next = tf.argmax(current_q_batch, axis=1)
        max_action_next = tf.reshape(max_action_next, shape=(-1, 1))
        # Create a mask so we only calculate loss on the updated Q-values
        target_q_batch = self.model_target.predict(state_next_sample).squeeze()
        # Q value = reward + discount factor * expected future reward

        target_q_batch = rewards_sample + GAMMA * tf.squeeze(tf.gather_nd(target_q_batch, max_action_next, batch_dims=1))
        # If final frame set the last value to -1
        target_q_batch = target_q_batch * (1 - done_sample) - done_sample

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)
            q_values = tf.squeeze(q_values, [1])

            # Apply the masks to the Q-values to get the Q-value for action taken
            action_sample = tf.reshape(action_sample, shape=(-1, 1))
            q_action = tf.gather_nd(q_values, action_sample, batch_dims=1)
            
            # Calculate loss between new Q-value and old Q-value
            loss = mean_squared_error(target_q_batch, q_action)
            # loss = h(target_q_batch, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

def cartpole():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DDQNSolver(observation_space, action_space)
    run = 0
    step_done = 0
    while True:
        run += 1
        observation, info = env.reset()
        observation = np.array(observation)
        observation = np.reshape(observation, [1, observation_space])
        step = 0
        while True:
            step += 1
            step_done += 1
            env.render()
            action = dqn_solver.act(observation)
            observation_next, reward, terminated, truncated, info = env.step(action)
            reward = reward if not terminated else -reward
            observation_next = np.reshape(observation_next, [1, observation_space])
            dqn_solver.remember(observation, action, reward, observation_next, terminated)
            observation = observation_next
            loss = dqn_solver.experience_replay()

            dqn_solver.exploration_rate = EXPLORATION_MIN + (EXPLORATION_MAX - EXPLORATION_MIN) * \
                                          math.exp(-1. * step_done / EXPLORATION_DECAY)

            if terminated:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break
        
        # Update the the target network with new weights
        if run % UPDATE_TARGET_FREQUENCY == 0:
            dqn_solver.model_target.set_weights(dqn_solver.model.get_weights())

        
        
cartpole()