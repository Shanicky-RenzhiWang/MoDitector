# MoDitector

This is the implement for the paper "MoDitector: Module-Directed Testing for Autonomous Driving Systems"

## Environment setup

### clone the repo

Clone this project with

    git clone --recurse-submodules git@github.com:Shanicky-RenzhiWang/MoDitector.git

### Setup conda environment

We have already export the running environment with `environment.yaml`, please excute

    conda env create -f environment.yaml

All the steps below is under this conda environment

    conda activate moditector

### install the dependency of Pylot

To running pylot in the experiment, excute

    cd ads/system/Pylot_project
    bash install.sh
    cd ../../..

Due to some python libs version confliction, please mannually install follow libs after pylot install

    pip install tensorflow-gpu==2.5.1
    pip install numpy==1.22

Some error in red may print, but don't be worried, it will effect nothing for Moditector.

## Run

We have already provided scripts to run the code in the `scripts` folder.
For example, you can use

    zsh scripts/fuzzing/pylot/highway_enter.sh

to run the experiment of highway_enter scenario.

Then you will reproduce the experiments.


Note that, if you would like to run the experiment without perception module, please edit the yaml file line 
`ori_config_target_agent` with add `_gt_perception` in the filename