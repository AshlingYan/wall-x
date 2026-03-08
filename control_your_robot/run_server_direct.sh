#!/bin/bash
cd /data/vla/wall-x/control_your_robot
export PYTHONPATH="/data/vla/wall-x/control_your_robot:/data/vla/wall-x"
/data/miniconda3/envs/wallx/bin/python scripts/server_wallx_fast.py
