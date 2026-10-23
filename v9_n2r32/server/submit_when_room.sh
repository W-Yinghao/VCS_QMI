#!/usr/bin/env bash
# Wait until this user's normal-QOS job count is below 28 (cap shared with other agents), then sbatch "$@".
until [ "$(squeue -u "$USER" -h -o %q | grep -c normal)" -lt 28 ]; do sleep 60; done
sbatch "$@"
