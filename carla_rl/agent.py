import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam



class DQNAgent:


    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.001
        self.model = self.build_model(state_size, action_size)
        self.target_model = self.build_model(state_size, action_size)
        self.update_target_model()

    '''Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    '''

    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))


    def build_model(self, state_size, action_size):
        # Limit GPU memory allocation to let game engine run
        # config = tf.ConfigProto(
        #     device_count={'GPU': 0}
        # )
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2

        set_session(tf.Session(config=config))

        model = Sequential()
        # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=state_size))
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(90, 160, 1)))
        # model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(45, 80, 3)))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(Flatten())
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=self.huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def add_batch_channel(self, state):
        if len(state.shape) == 2:
            return state[np.newaxis, :, :, np.newaxis]
        if len(state.shape) == 3:
            return state[np.newaxis, :, :, :]
        return state


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self.add_batch_channel(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = self.add_batch_channel(state)
            next_state = self.add_batch_channel(next_state)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)
