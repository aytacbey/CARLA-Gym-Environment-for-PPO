# CARLA-Gym-Environment-for-PPO
# Introduction
Salam (Peace)! 
For those who wants to turn CARLA 3D simulator into a gym-like environment in order to use CARLA's powerful game engine and python API without need to have massive computational power. The other reason was to enable the state-of-the-art Deep RL algorithm Proximal Policy Optimization (PPO) for people who wants to build a project in this field. Gym-like enviorenment was based on @CARLA 's work called [no_rendering_mode.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/no_rendering_mode.py) which turns the 3D simulator to a 2D map with all the actors in it. 
Secondly PPO was based on @ericyangyu 's work called [PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners). I would like to thank both parties for their outstanding work.

# How it works?
First of al it's recommended to create an virtual environment for this repo in order to not collide with pre-existing packages or even python version. There are numerous tutorials for virtual environments and how to activate them. In short below commands should be sufficient.
```
python3 -m venv /path/to/new/virtual/environment/my_env
source my_env/bin/activate
```
From this point onwards install the required dependencies to this virtual environment and run the repo from this virtual environment. 

Then run the CARLA simulator (linux).
```
./CarlaUE4.sh
```
If performance of your PC is an issue then the following command might help.
```
./CarlaUE4.sh -quality-level=Low
```
Last run the [main.py](https://github.com/aytacbey/CARLA-Gym-Environment-for-PPO/blob/main/main.py) either from IDE or kernel. I highly recommend to use kernel. For this 
```
cd path/to/repository python main.py
```

 
 
