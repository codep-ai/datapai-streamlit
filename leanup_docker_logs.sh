#!/bin/bash

# Truncate Docker container logs larger than 1GB

LOG_DIR="/var/lib/docker/containers"
SIZE="+1G"

find "$LOG_DIR" -type f -name "*-json.log" -size $SIZE -print -exec truncate -s 0 {} \;

