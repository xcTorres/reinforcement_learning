from array import array
import random
from xml.sax.saxutils import prepare_input_source
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
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995



optimizer=Adam(lr=LEARNING_RATE)

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))


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

        future_rewards = self.model.predict(state_next_sample).squeeze()
        # print(f'future_rewards: {future_rewards}')
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + GAMMA * tf.reduce_max(future_rewards, axis=1)
        # print(f'updated_q_values: {updated_q_values}')

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.action_space)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)
            
            q_values = tf.squeeze(q_values, [1])
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_max((tf.multiply(q_values, masks)), axis=1)
            
            # # Calculate loss between new Q-value and old Q-value
            loss = mean_squared_error(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def cartpole():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        observation, info = env.reset(seed=42)
        observation = np.array(observation)
        observation = np.reshape(observation, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = dqn_solver.act(observation)
            observation_next, reward, terminated, truncated, info = env.step(action)
            reward = reward if not terminated else -reward
            observation_next = np.reshape(observation_next, [1, observation_space])
            dqn_solver.remember(observation, action, reward, observation_next, terminated)
            observation = observation_next
            if terminated:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()

cartpole()