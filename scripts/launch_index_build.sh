#!/usr/bin/env bash
# Launcher for index build — runs via nohup, writes PID file
cd /home/brooklyn/splicing-ml-project
nohup bash scripts/03a_build_index.sh > scripts/index_build.log 2>&1 &
echo $! > scripts/index_build.pid
echo "Launched with PID: $(cat scripts/index_build.pid)"
