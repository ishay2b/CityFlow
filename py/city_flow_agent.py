import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import random 
import logging 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import gym


import numpy
import cityflow
import pandas as pd
import os
import csv
import json
from pathlib import Path as _P
import numpy as np
from numpy import array as _A
from timeit import default_timer
from ctypes import cdll, c_void_p, c_long, c_char_p
from matplotlib import pylab as plt
from pandas import DataFrame as _D
from pandas import json_normalize
import gym
import sys
import cv2

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam


PROJECT_ROOT =  _P(os.path.abspath(__file__)).parents[1]
PY_ROOT = PROJECT_ROOT / "py"
print("PY_ROOT", PY_ROOT)
sys.path.append(str(PY_ROOT))
os.chdir(PROJECT_ROOT)
print(f"Chdir {PROJECT_ROOT}")


config_path = PROJECT_ROOT / "data/esquare3/config_engine.json"


class CityFlowAgent(gym.Env):
    def __init__(self, config_path=config_path, mode='train'):
        
        work_dir = _P(config_path).parents[0]
        self.mode = mode
        self.last_action = None
        
        self.config = config = json.load(open(config_path, 'rt'))
        self.thread_num = config.get("thread_num", 4)
        
        roadnetFile = _P(config['dir']) / config['roadnetFile']
        self.roadnet = roadnet = json.load(roadnetFile.open('rt'))
        self.intersections = json_normalize(roadnet['intersections'])
        self.roads = roads = json_normalize(roadnet['roads'])
        self.intersections = self.intersections[~self.intersections['virtual']]  # Filter out virtuals
        
        self.lns = [len(row['trafficLight.lightphases']) for i, row in self.intersections.iterrows()]
        l = list()  # Read all points to get bounding box.
        for i, (p1, p2) in roads.points.iteritems(): l.extend((p1, p2)) 
        l = _D(l)
        self.min_xy = _A(l['x'].min(), l['y'].min())
        self.max_xy = _A(l['x'].max(), l['y'].max())
        self.MAT_SIZE = (self.max_xy - self.min_xy).max()
        self.range_1 = self.MAT_SIZE / (self.max_xy - self.min_xy)
        self.image = np.zeros((self.MAT_SIZE, self.MAT_SIZE), dtype=np.uint8)

        self.input_shape = (config['WINDOW_LENGTH'], self.MAT_SIZE, self.MAT_SIZE)

        num_phases = self.lns[0]
        self.action_space = gym.spaces.Discrete(num_phases)
        self.observation_space = gym.spaces.Dict({
            'phase':gym.spaces.Box(low=0, high=1, shape=(1,1), dtype=np.float32),
            'vehicle':gym.spaces.Box(low=0, high=14, shape=(1, self.MAT_SIZE, self.MAT_SIZE), dtype=np.float32)})
        #self.seed(seed, state)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.count_from_last_bump = 0
        self.total_num_bumps = 0
        self.curr_step = -1

        DIR = _P(config['dir'])
        #archive_dump = _P(config['dir']) / config['archive_dump']
        self.num_sim_loops = config.get('num_sim_loops', 500)
        self.output_dict_path = _P(config['dir']) / "summary.json"
        roudnet_output = DIR/ config['roadnetLogFile']
        replayLogFile = DIR/ config['replayLogFile']
        
        weights_filename = f"dqn_{config['WINDOW_LENGTH']}_{self.MAT_SIZE},{self.MAT_SIZE}_{mode}_weights.h5f"
        
        self.weights_filename = work_dir / weights_filename
        self.checkpoint_weights_filename = work_dir / ('checkpoint_{step}_' + weights_filename)

        if mode == 'train':
            self.config['saveReplay'] = False
            self.work_config_path = work_dir / "generated_train_config.json"            
        else:
            self.work_config_path = work_dir / "generated_predict_config.json"            
            self.config['saveReplay'] = True
                    
            try:
                os.remove(str(roudnet_output))
                print("Removed previous log file %s" % roudnet_output)
            except BaseException as e:
                print(str(e), roudnet_output)
            try:
                os.remove(str(replayLogFile))
                print("Removed previous log file %s" % replayLogFile)
            except BaseException as e:
                print(str(e), replayLogFile)
        self.eng = cityflow.Engine(str(self.work_config_path), thread_num=self.thread_num)
        
        json.dump(self.config, self.work_config_path.open('wt'))


    def step(self, action):
        self.eng.next_step()
        self.curr_step += 1
        #print(f"{self.curr_step}, Action {action}")
        if self.curr_step % 10 == 0:
            self.eng.bump_phase()
        if action != self.last_action and self.mode == 'predict':
            self.last_action = action
            print(f"{self.curr_step} Action changed to {action}")
        self.eng.set_tl_phase("intersection_1_1", action)
        self.count_from_last_bump = 0
        self.total_num_bumps += 1
    
        reward = self._get_reward()
        ob = self._get_state()
        done = (self.curr_step % 500) == (500-1)
        return ob, reward, done, dict(total_num_bumps=self.total_num_bumps)

    def run_baseline(self):
        ''' baseline '''
        action = 0
        acc = 0.0
        n = 0
        for i in range(self.config['steps_per_episode']):
            ob, reward, did_finish, info = env.step(action)
            acc += reward
            n += 1
            if i %  100 == 0:
                self.eng.bump_phase()
                sys.stdout.write(f"\r   {i}   , out of {self.config['steps_per_episode']}")
        return dict(n=n, acc=acc, reward=acc/n, steps_per_episode=self.config['steps_per_episode'])

    def _get_reward(self):
        return self.eng.get_average_vehicles_speed()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.count_from_last_bump = 0
        self.total_num_bumps = 0
        self.curr_step = -1
        self.curr_episode += 1
        self.eng.reset()
        return self._get_state()

    def render(self, mode="human", close=False):
        cv2.imshow("AA", self.image)
        cv2.waitKey(10)
        #plt.imshow(self.image)
        #display(plt.gcf())
        #display.clear_output(wait=True)

    def _get_state(self):
        """Get the observation."""
        locations = _A(self.eng.get_vehicles_location())
        self.image = self.quantize_list(locations)
        phases = self.eng.get_trafficLight_phases()
        position_speed_start = self.observation_space.spaces['vehicle'].low
        observation = {'phase':_A(phases), 'vehicle':self.image}	

        return observation

    
    def get_model(self):
        nb_actions = self.action_space.n
        inp = Input(shape=self.input_shape)
        X = Permute((2, 3, 1))(inp)
        X = Convolution2D(32, (8, 8), strides=(4, 4))(X)
        X = Activation('relu')(X)
        X = Convolution2D(64, (4, 4), strides=(2, 2))(X)
        X = Activation('relu')(X)
        X = Convolution2D(64, (3, 3), strides=(2, 2))(X)
        X = Activation('relu')(X)
        X = Flatten()(X)
        X = Dense(512)(X)
        X = Activation('relu')(X)
        x = Dense(nb_actions)(X)
        x = Activation('linear')(x)
        model = Model(inputs=inp, outputs=x)
        return model

    def quantize_list(self, v):
        mat = np.zeros((self.MAT_SIZE, self.MAT_SIZE), np.uint8)
        qv = ((v-self.min_xy)*self.range_1).astype(np.int8)
        qv = np.maximum(0, qv)
        qv = np.minimum(self.MAT_SIZE - 1, qv)
        for p in qv:
            mat[p[0], p[1]] += 1
        return mat


from rl.core import Processor


class MultiInputProcessor(Processor):
    """Converts observations from an environment with multiple observations for use in a neural network
    policy.
    In some cases, you have environments that return multiple different observations per timestep 
    (in a robotics context, for example, a camera may be used to view the scene and a joint encoder may
    be used to report the angles for each joint). Usually, this can be handled by a policy that has
    multiple inputs, one for each modality. However, observations are returned by the environment
    in the form of a tuple `[(modality1_t, modality2_t, ..., modalityn_t) for t in T]` but the neural network
    expects them in per-modality batches like so: `[[modality1_1, ..., modality1_T], ..., [[modalityn_1, ..., modalityn_T]]`.
    This processor converts observations appropriate for this use case.
    # Arguments
        nb_inputs (integer): The number of inputs, that is different modalities, to be used.
            Your neural network that you use for the policy must have a corresponding number of
            inputs.
    """
    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, state_batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        if hasattr(state_batch, 'shape'):
            if state_batch.shape >= (1,1):
                if isinstance(state_batch[0][0], dict):
                    return self.handle_dict(state_batch)
                if state_batch[0][0].ndim == 0:
                    if isinstance(state_batch[0][0].item(), dict):
                        return self.handle_dict(state_batch)
        for state in state_batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        return [np.array(x) for x in input_batches]

    def handle_dict(self,state_batch):
        """Handles dict-like observations"""

        names = state_batch[0][0].keys()
        ordered_dict = dict()
        for key in names:
            dim = len(state_batch[0][0][key].shape)
            order_dim = state_batch.shape
            for dim_count in range(dim):
                order_dim = order_dim + (state_batch[0][0][key].shape[dim_count],)
            order = np.zeros(order_dim)

            for idx_state, state in enumerate(state_batch):
                for idx_window in range(state_batch.shape[1]):
                    for i in range(order.shape[2]):
                        if not len(state_batch[idx_state][idx_window]) == self.nb_inputs: 
                            raise AssertionError()
                        order[idx_state, idx_window, i] = state_batch[idx_state][idx_window][key][i]
            ordered_dict[key] = order

        return ordered_dict

if __name__ == '__main__':
    start_time = default_timer()
    self = env = CityFlowAgent(config_path)
    eng = env.eng

    verbose_counter = self.config.get('verbose_counter', 100)

    from keras.models import Sequential, Model
    from keras.layers import Input, Flatten, LeakyReLU, Conv2D, MaxPooling2D, Activation, Dense, concatenate
    from keras.optimizers import Adam

    from rl.agents.dqn import DQNAgent
    from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
    from rl.memory import SequentialMemory

    ### Extract action space and shapes from the environment.
    nb_actions = env.action_space.n
    shape_phase = env.observation_space.spaces['phase'].shape
    shape_vehicle = env.observation_space.spaces['vehicle'].shape

    ### Phase model & input.
    model_phase = Sequential()
    model_phase.add(Flatten(data_format='channels_first', input_shape=shape_phase))
    model_phase_input = Input(shape=shape_phase, name='phase')
    model_phase_encoded = model_phase(model_phase_input)

    ### Vehicle model & input.
    model_vehicle = Sequential()
    model_vehicle.add(Conv2D(32, kernel_size=(4,4), strides=(1,1), data_format='channels_first', input_shape=shape_vehicle))
    model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_first'))
    model_vehicle.add(LeakyReLU())
    model_vehicle.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), data_format='channels_first'))
    model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_first'))
    model_vehicle.add(LeakyReLU())
    model_vehicle.add(Flatten(data_format='channels_first'))
    model_vehicle_input = Input(shape=shape_vehicle, name='vehicle')
    model_vehicle_encoded = model_vehicle(model_vehicle_input)

    ### Concatenation and final model. 
    conc = concatenate([model_phase_encoded, model_vehicle_encoded])
    hidden = Dense(128)(conc)
    hidden = LeakyReLU()(hidden)
    hidden = Dense(64)(hidden)
    hidden = LeakyReLU()(hidden)
    output = Dense(nb_actions, activation='linear')(hidden)
    model = Model(inputs=[model_phase_input, model_vehicle_input], outputs=output)
    model_path = "dqn_model.h5"
    try:
        model.load_weights(model_path)
        print(f"Success loading previous weights at {model_path}")
    except BaseException as e:
        print(f"Did not load previous weights due to {e}, {model_path}")

    ### Policy, Memory & Agent set-up.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.01, nb_steps=100000)
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, batch_size=64, gamma=.95, nb_steps_warmup=2000, target_model_update=.001)
    dqn.processor = MultiInputProcessor(2)
    dqn.compile(optimizer=Adam(lr=.001))

    ### Fit.
    hist = dqn.fit(env, nb_steps=200, verbose=1, log_interval=10)
    dqn.save_weights(model_path,  overwrite=True)
    print("Saved model to disk")

    test_env = CityFlowAgent(mode='predict', config_path=config_path)
    start_time = default_timer()
    dqn.test(test_env, nb_episodes=1, visualize=False) 
    print(f"\n Done testing inn {default_timer()-start_time} seconds")
