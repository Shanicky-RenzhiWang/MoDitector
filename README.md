# MoDitector

This is the implement for the paper "MoDitector: Module-Directed Testing for Autonomous Driving Systems"

## Environment setup

Our code is implemented in Pylot and Carla 0.9.10.1, please refer to [Pylot]{https://github.com/erdos-project/pylot.git} repo to complete the environment setup.

After setup, please clone this repo in the same directory with Pylot.

## Run

We have already provided scripts to run the code in the `scripts` folder.
For example, you can use

`sh scripts/data_collection/pylot/intersection_right.sh`

to run the experiment of intersection_right scenario.


Note that, if you would like to run the experiment without perception module, please edit the yaml file line 
`ori_config_target_agent` with add `_gt_perception` in the filename