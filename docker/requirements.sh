#!/bin/bash
source ./docker/format_print.sh export PYTHONWARNINGS="ignore:Unverified HTTPS request"
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

pip3 install -q --disable-pip-version-check opencv-python-headless==4.9.0.80
pip3 install -q --disable-pip-version-check colorlog==6.8.2 
pip install -q --disable-pip-version-check haystack-ai
pip install -q --disable-pip-version-check "datasets>=2.6.1"
pip install -q --disable-pip-version-check "sentence-transformers>=2.2.0"

pip install -q --disable-pip-version-check "huggingface_hub>=0.22.0"
pip install -q --disable-pip-version-check markdown-it-py mdit_plain pypdf
pip install -q --disable-pip-version-check gdown
# Remove Caches
rm -rf /var/lib/apt/lists/*
apt-get clean

printd -e "Done${REST}"