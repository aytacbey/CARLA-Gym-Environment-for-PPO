# CARLA-Gym-Environment-for-PPO
# Introduction
Salam (Peace)!
For those who wants to turn CARLA 3D simulator into a gym-like environment in order to use CARLA's powerful game engine and python API without need to have massive computational power. The other reason was to enable the state-of-the-art Deep RL algorithm Proximal Policy Optimization (PPO) for people who wants to build a project on this field. Gym-like enviorenment was based on @carla team's work called [no_rendering_mode.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/no_rendering_mode.py) which turns the 3D simulator to a 2D map with all the actors in it. 
Secondly PPO was based on @eyyu's work called [PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners). I would like to thank both parties for their contribution.

# How it works?
First of al it's recommended to create an virtual environment for this repo in order to not collide with pre-existing packages or even python version. There are numerous tutorials for virtual environments and how to activate them. In short below commands should be sufficient.
```python3 -m venv /path/to/new/virtual/environment
source <venv>/bin/activate```
First run the CARLA simulator ```./CarlaUE4.sh``` for linux.  
 
