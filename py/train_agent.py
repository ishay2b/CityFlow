
import numpy
import pandas as pd
import csv
import json
import numpy as np
from timeit import default_timer
from ctypes import cdll, c_void_p, c_long, c_char_p
from matplotlib import pylab as plt
from pandas import DataFrame as _D
from pandas import json_normalize
import gym
import sys
import os
from pathlib import Path as _P
from numpy import array as _A

PROJECT_ROOT =  _P(os.path.abspath(__file__)).parents[1]
PY_ROOT = PROJECT_ROOT / "py"
print("PY_ROOT", PY_ROOT)
sys.path.append(str(PY_ROOT))
from city_flow_agent import CityFlowAgent, PROJECT_ROOT

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K

import random 
import logging 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import gym

#from IPython import display
from PIL import Image

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam

from rl.callbacks import Callback
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = 'esqaure_rl'

mode = 'predict' if len(sys.argv) < 2 else sys.argv[1]
#env = gym.make(ENV_NAME)
config_path = PROJECT_ROOT / "data/esquare3/config_engine.json"
config = json.load(config_path.open('rt'))

env = CityFlowAgent(mode='train', config_path=config_path)

#np.random.seed(123)
#env.seed(123)
  
model = env.get_model()
model.summary()

weights_filename = env.weights_filename
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
memory = SequentialMemory(limit=1000000, window_length=env.config['WINDOW_LENGTH'])
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=150., value_min=0.0, value_test=.05, nb_steps=10000)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy, memory=memory, nb_steps_warmup=500, gamma=.9, target_model_update=1000, train_interval=100, delta_clip=1.)
dqn.compile(Adam(lr=.025), metrics=config['rl.metrics'])

did_load_weights = False

try:
    dqn.load_weights(weights_filename)
    did_load_weights = True
    print(f"Loaded weights from {weights_filename}")
except Exception as e:
    print(f"Did not load weights due to {e} , {weights_filename}")
    did_load_weights = False    
    
if not did_load_weights or mode=='train':
    callbacks = [ModelIntervalCheckpoint(env.checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    # dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)
    dqn.fit(env, callbacks=callbacks, nb_steps=50000, log_interval=5000, verbose=1)
    dqn.save_weights(weights_filename, overwrite=True)

    
env = CityFlowAgent(mode='predict', config_path=config_path)
if 0:
    for i in range(5000):
        ob, reward, did_finish, info = env.step(0)
        if i % 1000 == 1:
            print(i)
else:
    dqn.test(env, nb_episodes=1, visualize=False)
        
    
    

#dqn.test(env, nb_episodes=10, visualize=False, callbacks=[Render()])

