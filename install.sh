#!/bin/bash

#1) Create and activate environment
ENVS=$(conda info --envs | awk '{print $1}' )
if [[ $ENVS = *"sMartIoT"* ]]; then
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate sMartIoT
else
   echo "Creating a new conda environment for sMartIoT project..."
   #conda env create -n Face2PPG python=3.8
   conda env create -f environment.yml
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate sMartIoT
   #python setup.py install
   #exit
fi;

