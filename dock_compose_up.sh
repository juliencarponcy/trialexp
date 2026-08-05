#!/bin/bash

MACHINE_NAME=$(hostname)

if [ MACHINE_NAME = "jade" ]; then
    docker-compose --env-file linux.env up
elif [ MACHINE_NAME = "lapis" ]; then
    docker-compose --env-file linx_lapis.env up
else
    echo "Usage: ./run.sh [machine_a|machine_b]"
fi