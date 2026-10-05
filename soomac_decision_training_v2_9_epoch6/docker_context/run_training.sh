#!/bin/sh
set -eu
cd /app
exec python3 /app/train_in_container.py
