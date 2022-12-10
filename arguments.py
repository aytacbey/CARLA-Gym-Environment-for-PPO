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

    parser.add_argument('--mode', dest='mode', type=str, default='train')  # can be 'train' or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, 
                        default='actor_model.pth')  # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, 
                        default='critic_doel.pth')  # your critic model filename

    args_PPO = parser.parse_args()

    # Define arguments that will be received and parsed for CARLA
    argparser = argparse.ArgumentParser(
        description='CARLA No Rendering Mode Visualizer')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        default=None,
        help='start a new episode at the given TOWN')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='switch off server rendering')
    argparser.add_argument(
        '--show-triggers',
        action='store_true',
        help='show trigger boxes of traffic signs')
    argparser.add_argument(
        '--show-connections',
        action='store_true',
        help='show waypoint connections')
    argparser.add_argument(
        '--show-spawn-points',
        action='store_true',
        help='show recommended spawn points')
    
    # pygame.init()
    # Parse arguments
    args_env = argparser.parse_args()
    args_env.description = argparser.description

    return args_PPO, args_env, args_env.description
