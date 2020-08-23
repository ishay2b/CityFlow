from gym import spaces
from gym.utils import seeding
import traci
import numpy as np
from sumolib import checkBinary
from trips import generate, delete
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, LeakyReLU, Conv2D, MaxPooling2D, Activation, Dense, concatenate
from keras.optimizers import Adam

from rl_.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl_.multiInputProcessor import MultiInputProcessor

#from sumoEnv import SumoEnv

class SumoEnv(Env):

   def __init__(self, seed=None, state=None):
      self.action_space = spaces.Discrete(4)
      self.observation_space = spaces.Dict({'phase':spaces.Box(low=0, high=1, shape=(4,1,3), dtype=np.float32), 
                                            'vehicle':spaces.Box(low=0, high=14, shape=(29,29,2), dtype=np.float32)})
      self.seed(seed, state)

   def seed(self, seed, state):
      '''
      Returns:
         Float of the seed used.
      '''
      self.rng, seed = seeding.np_random(seed)
      if state != None:
         self.rng.set_state(state)
      return [seed]
      
   def reset(self):
      '''
      Returns:
         Nd.array of starting observation.
      '''
      traci.close()
      delete(prefix="trone")
      generate(prefix="trone", src=["src"+str(n) for n in range(5)], dst=["dst"+str(n) for n in range(9)], rng=self.rng, scale=(10,10))
      traci.start([checkBinary("sumo"), "-c", "trone.cfg"])

      phase_start = self.observation_space.spaces['phase'].low
      position_speed_start = self.observation_space.spaces['vehicle'].low
      observation = {'phase':phase_start, 'vehicle':position_speed_start}	

      return observation

   def step(self, action, observe=True):
      '''
      Args:
         action: the desired phase.
      Returns:
         observation: Dictionary of the observation space after the action has been taken.
         reward: Float of the reward the action taken has resulted in.
         done: Boolean indicating if the current episode is done.
      '''
      # Action-dependent phase changing
      desired_phase = action
      current_phase = traci.trafficlight.getPhase("tls_id")
      if desired_phase != current_phase:
         traci.trafficlight.setPhaseDuration("tls_id", 0)
      # Action-independent simulation tracking
      sim_info = self.trackSim(sec=3)
      reward = sim_info['added_disutility']
      # Action-dependent phase changing
      if desired_phase != current_phase:
         traci.trafficlight.setPhase("tls_id", desired_phase)
      # Episode end check
      done = self._elapsedTime() >= 3600
      # Observation
      position_speed_mat = self.getVehicleInfo()
      self.phase_hist = [desired_phase]+self.phase_hist[:2] 
      phase_hist = np.dstack(self.phase_hist)
      observation = {'phase':phase_hist, 'vehicle':position_speed_mat}

      return observation, reward, done, {}
    



### Extract action space and shapes from the environment.
nb_actions = env.action_space.n
shape_phase = env.observation_space.spaces['phase'].shape
shape_vehicle = env.observation_space.spaces['vehicle'].shape

### Phase model & input.
model_phase = Sequential()
model_phase.add(Flatten(data_format='channels_last', input_shape=shape_phase))
model_phase_input = Input(shape=shape_phase, name='phase')
model_phase_encoded = model_phase(model_phase_input)

### Vehicle model & input.
model_vehicle = Sequential()
model_vehicle.add(Conv2D(32, kernel_size=(4,4), strides=(1,1), data_format='channels_last', input_shape=shape_vehicle))
model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_last'))
model_vehicle.add(LeakyReLU())
model_vehicle.add(Conv2D(64, kernel_size=(4,4), strides=(1,1), data_format='channels_last'))
model_vehicle.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format='channels_last'))
model_vehicle.add(LeakyReLU())
model_vehicle.add(Flatten(data_format='channels_last'))
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

### Policy, Memory & Agent set-up.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.01, nb_steps=100000)
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, batch_size=64, gamma=.95, nb_steps_warmup=2000, target_model_update=.001)
dqn.processor = MultiInputProcessor(2)
dqn.compile(optimizer=Adam(lr=.001))

### Fit.
hist = dqn.fit(env, nb_episodes=200, verbose=0)

