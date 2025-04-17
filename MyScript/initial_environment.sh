#!/bin/bash

# Define mirror URL
MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# Create pip config directory if not exists
mkdir -p ~/.pip

# Write the pip.conf
cat <<EOF > ~/.pip/pip.conf
[global]
index-url = ${MIRROR_URL}
timeout = 60
EOF

pip install -e .
apt-get update && apt-get install -y vim
