"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

#import gym
import time
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    import csv
    def  __init__(self, policy_class, env, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
        """
		# Make sure the environment is compatible with our code
# 		assert(type(env.observation_space) == gym.spaces.Box)
# 		assert(type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

		# Extract environment information
        self.env = env

        self.obs_dim = env.observation_space.shape[1]
        self.act_dim = env.action_space.shape[1]
        print("&&&&&&& obs ve act dim", self.obs_dim, self.act_dim)
		 # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var) #diagonalleri oluşturdu
        print("cov_var",self.cov_var,"self.cov_mat",self.cov_mat)
		# This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            }

    def learn(self, total_timesteps):
        """
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
        import csv
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Collecting our batch simulations here
            (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens,
            batch_rews, batch_episodic_timesteps,batch_overall_timesteps, 
            batch_rews_to_store, ep_no, batch_accumulated_speed, 
            batch_total_overtakes, batch_total_collisions, batch_total_of_road,
            batch_aux_probs) = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
            i_so_far += 1
            
			# Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
            V_ini, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V_ini.detach()    
            V_ini = V_ini.detach()  
            # for i in range(len(A_k)):
            #      a_k = batch_rtgs[i] - V_ini.detach()[i] 
             #    print("advantage hsesaplaması ",a_k, batch_rtgs[i], V_ini.detach()[i])                          # ALG STEP 5
            A_k_test = A_k
			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            

            #string = "./baz_test/base_wo_vehicles" + str(i_so_far) + ".csv"
            # with open ("./baz_test/base_wo_vehicles_v3.csv","a",newline = "") as csvfile:

            with open ("./test_results.csv", "a", newline = "") as csvfile:
                writer = csv.writer(csvfile)
                print(batch_episodic_timesteps, batch_overall_timesteps)
                _obs = batch_obs.tolist()
                _act = batch_acts.tolist()
               # print(batch_acts,_act)
                # print("listeye .evrilmiş ",_obs)
                for (ep_tim,obs, tot_tim,rews ,acts,rtgs,speed, ovrtk_veh,
                     tot_col, tot_ofroad, aux_prob ) in zip(
                         batch_episodic_timesteps, _obs, 
                         batch_overall_timesteps, batch_rews_to_store, 
                         batch_acts, batch_rtgs, batch_accumulated_speed, 
                         batch_total_overtakes, batch_total_collisions, 
                         batch_total_of_road, batch_aux_probs):                    
                    writer.writerow([ep_tim,obs, tot_tim,rews ,acts,rtgs,
                                     speed, ovrtk_veh, tot_col, tot_ofroad, 
                                     aux_prob])
			# This is the loop where we update our network for some n epochs
            # PPO ALG STEP 6 & 7
            for _ in range(self.n_updates_per_iteration):      # epoch number 
				# Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs) # batch log probsu her epoch için basmak mantıklı değil, değişmiyor sonuçta

				# Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                aux_ratios = ratios
                print(len(A_k), "A_k nin boyutu", len(A_k.detach()))
                for i in range(len(A_k.detach())):
                    if batch_aux_probs[i] == 1 and A_k[i]< 0:
                        aux_ratios[i] = max(ratios[i], 1.2)
                surr3 = aux_ratios * A_k  

                # Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
                # Note to self: Mean of the total loss is calculated here.
                # Aux rew should be implemented here
                actor_loss_batch = -torch.min(surr2, surr3).detach()
                actor_loss = (-torch.min(surr2, surr3)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                print("actor lossu", actor_loss.detach())
                _actor_loss = (-torch.min(surr2, surr3)).mean().detach()
                _critic_loss = nn.MSELoss()(V, batch_rtgs).detach()                
                actor_loss_mean = [_actor_loss for i in range(len(surr1))]
                critic_loss_mean = [_critic_loss for i in range(len(surr1))]
				# Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

				# Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())
                
                _V = V.detach()
                _batch_log_probs = batch_log_probs.detach()
                _curr_log_probs = curr_log_probs.detach()
                _ratios = ratios.detach()
                _surr1 = surr1.detach()
                _surr2 = surr2.detach()
                _surr3 = surr3.detach()
                # with open ("./baz_test/base_wo_vehicles_bp_v3.csv","a",newline = "") as csvfile:
                with open ("./baz_test_yeni_parkur_v4/base_w_vehicles_bp_yeni_parkur_v5_aux.csv","a",newline = "") as csvfile:
                    writer = csv.writer(csvfile)

                    for (V, ep_t, batch_log, tot_time, Adv, cur_log, ratio, sur1, sur2, actor_loss_b, act_los_mean, cri_los_mean, v_ini, sur_3, aux_rat) in zip(_V, batch_episodic_timesteps,  _batch_log_probs, batch_overall_timesteps, A_k, _curr_log_probs, _ratios, _surr1, _surr2,actor_loss_batch, actor_loss_mean, critic_loss_mean, V_ini, _surr3, aux_ratios):                    
                        writer.writerow([V, ep_t, batch_log, tot_time, Adv, cur_log, ratio, sur1, sur2, actor_loss_b, act_los_mean, cri_los_mean, v_ini, sur_3, aux_rat])
                # with open ("./baz_test/base_wo_vehicles_ep_rews_v3.csv","a",newline = "") as csvfile: 
                with open ("./baz_test_yeni_parkur_v4/base_w_vehicles_ep_rews_yeni_parkur_v5_aux.csv","a",newline = "") as csvfile:
                    writer = csv.writer(csvfile)

                    for ep_rew in batch_rews:
                        for (ep_time,rew, adv, disc_r, V, v_init, tot_time) in zip(batch_episodic_timesteps,ep_rew, A_k_test, batch_rtgs, _V, V_ini, batch_overall_timesteps):                    
                            writer.writerow([ep_time, rew, adv.detach(), disc_r, V, v_init, tot_time])
                    
        # with open ("./baz_test/base_wo_vehicles.csv","w",newline = "") as csvfile:
            #     writer = csv.writer(csvfile)
            #     for row in batch_obs:                    
            #         writer.writerow(row) #[req_array her epoch için batch obs, act vb. yi kaydetme mantıklı olmayabilir ])
           # print("epoch ",surr1, " ",surr2, " ", actor_loss,"     ")
			# Print a summary of our training so far
            self._log_summary()

			# Save our model if it's time
            if i_so_far % self.save_freq == 0:
                print("saving actor and critic model")
                string_actor = ("./actor_"
                                + str(i_so_far) 
                                + "_" 
                                + str(self.avg_ep_rews) 
                                + ".pth")
                string_critic = ("./critic_" 
                                 + str(i_so_far) 
                                 + "_" 
                                 + str(self.avg_ep_rews) 
                                 + ".pth")               
                torch.save(self.actor.state_dict(), string_actor)
                torch.save(self.critic.state_dict(), string_critic)
                
    def rollout(self):
        """torch.tensor(self.observation_space, dtype=torch.float)
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_overall_timesteps = []
        batch_episodic_timesteps = []
        batch_rews_to_store = []
        batch_accumulated_speed = []
        batch_total_overtakes = []
        batch_total_collisions = []
        batch_total_of_road = []
        batch_aux_rewards = []
		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
        ep_rews = []
        ep_aux_rews = []     # 
        t = 0 # Keeps track of how many timesteps we've run so far this batch
        ep_no = 0
		# Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_no += 1
            ep_rews = [] # rewards collected per episode
    
			# Reset the environment. sNote that obs is short for observation. 
            """Bu noktada env reset edilip (tüm araçlar) başa dönülecek"""
            obs = self.env.reset()
            done = False
          #  print("PPO-reset sonrası")
			# Run an episode for a maximum of max_timesteps_per_episode timesteps
            """Episode uzunluğu burda belirlenecek"""
            for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
                # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                #     self.env.render()

                t += 1 # Increment timesteps ran this batch so far
				# Track observations in this batch
                batch_obs.append(obs)
                # print("toplam adım %s"%t,"episod adım%s"%ep_t)
				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
                """Her bir timestep için önce action belirleniyor ardından env bu action gönderiliyor"""
                # Action's are evaluated
                action, log_prob = self.get_action(obs)   
                # Action's are sent to the gym-env
                obs, rew, done, current_speed, total_overtakes,       \
                total_coll, total_of_road, aux_rew = self.env.step(action)

				# Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_aux_rews.append(aux_rew)
                batch_rews_to_store.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_overall_timesteps.append(t)    # total timestep in batch
                batch_episodic_timesteps.append(ep_t + 1) # episod timestepi
                batch_accumulated_speed.append(current_speed)
                batch_total_overtakes.append(total_overtakes)
                batch_total_collisions.append(total_coll)
                batch_total_of_road.append(total_of_road)
 

				# If the environment tells us the episode is terminated, break
                if done:
                    break
                    print("DONE - BREAK")
            print("BREAK")
			# Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_aux_rewards.append(ep_aux_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
        self.stored_batch_obs = batch_obs   
        batch_obs = torch.stack(batch_obs) #orj torch.tensor(batch_acts, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.stack(batch_log_probs) #orj torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs, batch_aux_probs =     \
            self.compute_rtgs(batch_rews, batch_aux_rewards)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        return (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens,
                batch_rews, batch_episodic_timesteps,batch_overall_timesteps, 
                batch_rews_to_store, ep_no, batch_accumulated_speed, 
                batch_total_overtakes, batch_total_collisions, 
                batch_total_of_road, batch_aux_probs)

    def compute_rtgs(self, batch_rews, batch_aux_rewards):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
        batch_rtgs = []
        batch_aux_probs = []
		# Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
               # print("*** rewards",ep_rews)
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        # Auxiliary reward is in the trial phase.
        # It's not part of the main PPO algorithm in any shape or form.
        for ep_aux_rews in reversed(batch_aux_rewards):
            end_point = 0
            start_point = 0
            for cnt, aux_rew in enumerate(reversed(ep_aux_rews)):
                # It's expected to have max one aux reward in each episode.
                if aux_rew == 1:
                    end_point = cnt + 100
                    start_point = cnt + 40
                if cnt < end_point and cnt > start_point:
                       
                    batch_aux_probs.insert(0, 1)            
                else:
                    batch_aux_probs.insert(0, 0) 
		# Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs, batch_aux_probs

    def get_action(self, obs):
        """
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
    #    print("mean=query")
		# Query the actor network for a mean action
        mean = self.actor(obs)
      #  print("mean=query - action üretiliyor")
		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
        action = dist.sample()

		# Calculate the log probability for that action
        log_prob = dist.log_prob(action)
       # print("self.cov_mat",self.cov_mat, "mean", mean, "dist",dist, "action", action, "log_prob", log_prob)
		# Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""

		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
        self.timesteps_per_batch = 400_800_000     # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 150_000   # Max number of timesteps per episode
        self.n_updates_per_iteration = 5           # Number of times to update actor/critic per iteration
        self.lr = 0.005                            # Learning rate of actor optimizer
        self.gamma = 0.95                          # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                            # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
        self.render = True                         # If we should render during rollout
        self.render_every_i = 10                   # Only render every n iterations
        self.save_freq = 5000                      # How often we save in number of iterations
        self.seed = None                           # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
        if self.seed != None:
			# Check if our seed is valid first
            assert(type(self.seed) == int)

			# Set the seed 
            torch.manual_seed(self.seed)
            #f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        self.avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(self.avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
        #flush=True)
        print(f"---------- Iteration #{i_so_far} ----------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"-------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
