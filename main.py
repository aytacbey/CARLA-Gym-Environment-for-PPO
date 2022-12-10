"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

"""
Created on Sat May 14 18:13:38 2022
    
@author: Aytaç Kasımoğlu
"""
import os
import argparse
from threading import Thread
from CarlaEnv import InputControl, HUD, World
import sys
import torch
import pygame
from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy


COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)
    
COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)


env= None

def train(env, hyperparameters, actor_model, critic_model):
    """
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
    print(f"Training", flush=True)

	# Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:

        print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
    """
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
    """
    print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

	# Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)

def main(args_env):
    
    """
		The main function to run.

		Parameters:
			args_env - the arguments parsed from command line

		Return:
			None
    """
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
				'timesteps_per_batch': 4000_000,        # 2048
				'max_timesteps_per_episode': 2100,  # 1800
				'gamma': 0.99, 
				'n_updates_per_iteration': 1,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	# Train or test, depending on the mode specified
    if args_PPO.mode == 'train':
        print("train mode")
        train(env=world, hyperparameters=hyperparameters, 
              actor_model=args_PPO.actor_model, 
              critic_model=args_PPO.critic_model)
    else:
        test(env=env, actor_model=args_PPO.actor_model)


def pygame_loop(world,hud,input_control,display,clock, COLOR_ALUMINIUM_4):
    clock.tick_busy_loop(60)
    
    # Tick all modules
    world.tick(clock)
    input_control.tick(clock)    
    hud.tick(clock)

    # Render all modules
    display.fill(COLOR_ALUMINIUM_4)
    world.render(display)
    hud.render(display)
    input_control.render(display)

    pygame.display.flip()
    
    
if __name__ == '__main__':

    """Parses the arguments received from commandline and runs the game loop"""

    args_PPO, args_env, args_env.description = get_args() # Parse arguments from command line
    args_env.width, args_env.height = [int(x) for x in args_env.res.split('x')]
    pygame.init()    
    
    try:

        display = pygame.display.set_mode(
        (args_env.width, args_env.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
        input_control = InputControl("TITLE_INPUT")    # input control instance
        hud = HUD("TITLE_HUD", args_env.width, args_env.height)    
        world = World("TITLE_WORLD", args_env, timeout=2000.0) 
        
        # For each module, assign other modules that are going to be used inside that module
        input_control.start(hud, world)  # input control requires hud and world
        hud.start()                      # hud started        

        # Game loop        
        clock = pygame.time.Clock()
        clock.tick_busy_loop(60)
      
        world.start(hud, input_control, display, clock, COLOR_ALUMINIUM_4)   
    

        """Before starting the PPO algorithm let's render the first scene"""
        # Tick all modules
        world.tick(clock)
        hud.tick(clock)
        input_control.tick(clock)

        # Render all modules
        display.fill(COLOR_ALUMINIUM_4)
        world.render(display)
        hud.render(display)
        input_control.render(display)

        pygame.display.flip()
        main(args_env)


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    
    finally:
        if world is not None:
            print("destroy MAIN FINAL")
            world.destroy()

