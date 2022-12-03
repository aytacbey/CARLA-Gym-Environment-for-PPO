"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

def get_args():
	"""
		Description:
		Parses arguments at command line.

		Parameters:
			None

		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='/home/aytac/Masaüstü/00_Projeler/CARLA/PythonAPI/examples/ppo_actor_10_2_83.pth')     # your actor model filename
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='/home/aytac/Masaüstü/00_Projeler/CARLA/PythonAPI/examples/ppo_critic_10_2_83.pth')   # your critic model filename

	args2 = parser.parse_args()

	return args2
