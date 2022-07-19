import gym
from gym import spaces
import numpy as np 
import matplotlib.pyplot as plt

class myEnv_Daily(gym.Env):
	
	metadata = {'render.modes':['human']}
	
	def __init__(self, N, initial_state, alpha1, alpha2, alpha_linear):
		super(myEnv_Daily, self).__init__()
		
		# Environment hyperparamters
		self.N = N
		self.initial_state = initial_state
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.alpha_linear = alpha_linear
		
		# SEIR parameteres
		self.epsilon = 0.2
		self.gamma = 0.2
		self.alpha = 0.006
		self.threshold_highdanger = self.N * 100/100000  # highrisk transmission: >100 new cases in the last 7 days per 100k people

		
		# Action space
		self.action_space = spaces.Discrete(3)
		
		# Observation space (s, e, i, r, d, 6days ago's new cases, ..., today's new cases)
		low = 0
		high = 1
		self.observation_space = spaces.Box(low=low, 
											high=high, 
											shape=(12,), dtype=np.float16)

		# state, max ep length, episode length tracker
		self.state = None
		self.max_episode_length = np.random.randint(250,550)
		self.steps_remaining = self.max_episode_length
		
	def seir(self, rt):
		# get today's input state (yestserday's seir outputs)
		normalized_state = self.state

		# un-normalized them to be readable
		readable_state = self.unnormalize(normalized_state)
		sus, exp, inf, rec, dea, day7, day6, day5, day4, day3, day2, day1 = readable_state

		# get beta parameter
		beta = self.gamma * rt
		
		# get next state
		sus2 = sus - np.round(beta*inf*sus/self.N) 
		exp2 = exp - np.round(self.epsilon*exp) + np.round(beta*inf*sus/self.N)
		inf2 = inf + np.round(self.epsilon*exp) - np.round(self.gamma*inf) - np.round(self.alpha*inf)
		rec2 = rec + np.round(self.gamma*inf)
		dea2 = dea + np.round(self.alpha*inf)
	
		# get today's new cases
		newcases = np.round(self.epsilon*exp)

		# normalize the readable states, and return
		return_state = np.array([sus2, exp2, inf2, rec2, dea2, day6, day5, day4, day3, day2, day1, newcases])
		return_state = self.normalize(return_state)

		return return_state

	def step(self, action):
		#print(self.state)
		rt = 0
		reward1 = 0
		reward2 = 0
		rewardt = 0
		done = False

		# convert actions into r(t) effective reproduction number
		# and compute reward 1, open-ness
		if action==0:
			rt = 0.7
			reward1 = 0 ### no reward/penalty for lockdown
		elif action==1:
			rt = 1.2
			reward1 = self.alpha1 ### positive alpha1 reward for small-opening
		elif action==2:
			rt = 1.5
			reward1 = self.alpha2 ### positive alpha2 reward (alpha2>alpha1) for big-opening

		
		# go through SEIR transition 
		self.state = self.seir(rt)

		# take resulting state and unnormalized, because we need new-cases data
		unnormalized_state = self.unnormalize(self.state)
		sus, exp, inf, rec, dea, day7, day6, day5, day4, day3, day2, day1 = unnormalized_state

		# compute reward 2, new cases
		newcases_7days = day7 + day6 + day5 + day4 + day3 + day2 + day1
		#threshold_highdanger = threshold_highdanger*0.9 # lower a little bit for safety measure

		if newcases_7days < self.threshold_highdanger:
			reward2 = 0 ### no cost if below threshold 
		else: 
			reward2 = - (newcases_7days/self.threshold_highdanger) * self.alpha1 ### cost associated with threshold. 

		# compute total reward
		rewardt = reward1 + self.alpha_linear*reward2
		
		# check done 
		# done case 1 (when it goes to shit, then might as well restart, there is no point to continue episode)
		if newcases_7days > self.N/10: 
			done = True        
		# done case 2 (when max ep length is reached)
		elif self.steps_remaining == 0:
			done = True
		#elif self.steps_remaining > 0:
		
		# take a step
		self.steps_remaining = self.steps_remaining - 1
		
		
		return self.state, rewardt, done, {}
		
	
	def normalize(self, orig_state):
		normalized_state = np.zeros(12)

		# normalize S,E,I,R,D by N because none of then will ever surpass N
		normalized_state[:5] = orig_state[:5] / self.N 
		# nomalized new case each day by (self.threshold_highdanger*6)
		normalized_state[5:] = orig_state[5:] / self.N

		return normalized_state

	def unnormalize(self, orig_state):
		unnormalized_state = np.zeros(12)

		# unnormalize S,E,I,R,D by N because none of then will ever surpass N
		unnormalized_state[:5] = orig_state[:5] * self.N 
		# unnomalized new case each day by (self.threshold_highdanger*6)
		unnormalized_state[5:] = orig_state[5:] * self.N

		return unnormalized_state

	def reset(self, perform=False):
		# reset initial state
		self.state = self.normalize(self.initial_state)

		if perform == True:
			self.max_episode_length = 365
			#print("perform is true!")
		else:
			# reset episode length, by random
			self.max_episode_length = np.random.randint(250,550)
		# reset step counter to that length
		self.steps_remaining = self.max_episode_length

		return self.state
	
	def render(self, mode='human'):
		day = self.max_episode_length - self.steps_remaining
		print('Day {}: {}'.format(day, self.state))

