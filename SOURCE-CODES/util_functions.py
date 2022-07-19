import gym
from gym import spaces
import numpy as np 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

### this function is smply SEIR, 
### used to set up the start of the epidemic 
def seir_beforeRL(initial_state, r0, N): 
	# SEIR parameteres
	epsilon = 0.2
	gamma = 0.2
	alpha = 0.006    
	beta = gamma * r0
	
	# get today's input state (yestserday's seir outputs)
	sus, exp, inf, rec, dea, day7, day6, day5, day4, day3, day2, day1 = initial_state
	
	# get next state
	sus2 = sus - np.round(beta*inf*sus/N)
	exp2 = exp - np.round(epsilon*exp) + np.round(beta*inf*sus/N)
	inf2 = inf + np.round(epsilon*exp) - np.round(gamma*inf) - np.round(alpha*inf)
	rec2 = rec + np.round(gamma*inf)
	dea2 = dea + np.round(alpha*inf)
	
	# get today's new cases
	newcases = np.round(epsilon*exp)
	
	return np.array([sus2, exp2, inf2, rec2, dea2, 
					 day6, day5, day4, day3, day2, day1, newcases])


### input: matrix of states over time of shape (time, 12)
### output: array of "new cases in the last 7 days" of shape (time, 1)
def get_sum7days(state_matrix):
	return np.sum(state_matrix[:, 5:12], axis=1)


### function to make the final plot that includes preRL+RL
def make_final_plot(total_preRL_days, total_actions, total_observations, newcases_7days, threshold_highdanger, num_actions=3):
	figure(figsize=(10, 6), dpi=100)

	plt.plot(newcases_7days, label="new cases in the last 7 days")
	plt.plot(total_actions*100*2/(num_actions-1), label ="actions") # multiplied actions 0,1,2 by 100 for the sole purpose of being able to view actions clearer on the graph

	plt.plot(threshold_highdanger*np.ones(total_observations.shape[0]), 
			 label="high-risk threshold of new cases in the last 7 days", 
			 linestyle="--")
	plt.axvline(x=30, color='grey', label="lockdown begins")
	plt.axvline(x=total_preRL_days, color='purple', 
				label="rl agent actions begins")

	plt.xlabel('day')
	plt.ylabel('# of people')
	plt.plot()
	plt.legend()


### simulates preRL
def simulate_preRL(N, E_init, threshold_highdanger, total_preRL_days):
	state =  np.array([N - E_init, E_init, 0, 0, 0, 
						   0,0,0,0,0,0,0])
	preRL_states = np.zeros((total_preRL_days, 12))

	# run pre-RL
	for i_day in np.arange(total_preRL_days):
		if i_day < 30:
			r0 = 3 
		else:
			r0 = 0.7 
		state = seir_beforeRL(initial_state=state, r0=r0, N=N)
		preRL_states[i_day] = state

	final_state = state
	return final_state, preRL_states



### to test performance of 2week envs ### 
def test_2week(env, agent):
	observations = []
	actions = []
	agent.initialize(perform=True)

	initial_observation = agent.s_0
	#observations.append(unnormalize(initial_observation, env.N, env.threshold_highdanger))

	for time in range(365):
	#print('Time: {}'.format(time))
		if (time % 14 ==0): # if its been 2 weeks, make an action 
			action, bulk_states, r, done = agent.perform_step()
			#print("action: {}".format(action))
			
			for i in range(len(bulk_states)):
					actions.append(action)
					observations.append(unnormalize(bulk_states[i], env.N, env.threshold_highdanger))
					#print(bulk_states[i])

	print('Total Reward: {}'.format(r))

	return actions, observations


### to test performance of VARY-FREQ envs ### 
def test_varyfreq(env, agent, frequency):
	observations = []
	actions = []
	agent.initialize(perform=True)

	initial_observation = agent.s_0
	#observations.append(unnormalize(initial_observation, env.N, env.threshold_highdanger))

	for time in range(365):
	#print('Time: {}'.format(time))
		if (time % frequency ==0): # if its been x days, make an action 
			action, bulk_states, r, done = agent.perform_step()
			#print("action: {}".format(action))
			
			for i in range(len(bulk_states)):
					actions.append(action)
					observations.append(unnormalize(bulk_states[i], env.N, env.threshold_highdanger))
					#print(bulk_states[i])

	print('Total Reward: {}'.format(r))

	return actions, observations


### combines preRL+RL, usually for graphing purposes 
def combine_preRL_and_RL(preRLinfo, RLinfo):
	# make sure everything is in numpy form
	preRLinfo = np.asarray(preRLinfo)
	RLinfo = np.asarray(RLinfo)
	# concatenate
	combined_info = np.concatenate((preRLinfo, RLinfo))

	return combined_info

### maybe depreciated, for daily envs 
def test_daily(env, agent):
	observations = []
	actions = []
	agent.initialize(perform=True)
	initial_observation = agent.s_0
	#print('Initial: {}'.format(observation))
	#observations.append(initial_observation)

	for time in range(365):
		action, s_1, r, done = agent.perform_step()
		actions.append(action)
		observations.append(unnormalize(s_1, env.N, env.threshold_highdanger))
		if done:
			print("Episode finished after {} timesteps".format(time+1))
			break
	print('Total Reward: {}'.format(r))

	return actions, observations



### to unnormalize state-vectors. 
### input: a 12-dim state vector
def unnormalize(orig_state, N, threshold_highdanger):
	unnormalized_state = np.zeros(12)
	# unnormalize S,E,I,R,D by N because none of then will ever surpass N
	unnormalized_state[:5] = orig_state[:5] * N 
	# unnomalized new case each day by (self.threshold_highdanger*6)
	unnormalized_state[5:] = orig_state[5:] * N
	return unnormalized_state
