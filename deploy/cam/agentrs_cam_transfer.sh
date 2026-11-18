#!/bin/bash
# Waits for the main CAM run, then runs the bizdev learning-speed transfer test (scratch vs pretrained).
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
exec bash /workspace/agentrs-cam/scripts/cam_transfer.sh 2>&1
