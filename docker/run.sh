#!/bin/bash
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# --------------------------------------------------------
# Sub function
function print_magic(){
	info=$1
	magic=$2
	echo ""
	if [[ $magic = true ]];then
		echo -e $info | boxes -d dog -s 80x10
	else
		echo -e $info
	fi
	echo ""
}

# ---------------------------------------------------------
# Color ANIS
RED='\033[1;31m';
BLUE='\033[1;34m';
GREEN='\033[1;32m';
YELLOW='\033[1;33m';
CYAN='\033[1;36m';
NC='\033[0m';

# ---------------------------------------------------------
function init_word(){
  echo -e "${RED}"
  echo "
  ====================================================

              ██████╗░░█████╗░░██████╗░
              ██╔══██╗██╔══██╗██╔════╝░
              ██████╔╝███████║██║░░██╗░
              ██╔══██╗██╔══██║██║░░╚██╗
              ██║░░██║██║░░██║╚██████╔╝
              ╚═╝░░╚═╝╚═╝░░╚═╝░╚═════╝░ 
	
  ====================================================
  "
  echo -e "${NC}"
}

# ---------------------------------------------------------
init_word

# ---------------------------------------------------------
# Install jq
echo -e "${YELLOW}"
echo "----- Installing JQ -----"
echo -e "${NC}"

if ! type jq >/dev/null 2>&1; then
    sudo apt-get install -y jq
else
    echo 'The jq has been installed.';
fi


# --------------------------------------------------------
# Parse information from configuration
CONF="./docker/configuration/version.json"
USER=$(cat ${CONF} | jq -r '.USER')
BASE_NAME=$(cat ${CONF} | jq -r '.PROJECT')
TAG_VER=$(cat ${CONF} | jq -r '.VERSION')

#Get mount folder (dataset,pretrain model)
MCONF="./docker/configuration/mount.json"
DATASET=$(cat ${MCONF} | jq -r '.dataset')
PRETRAIN=$(cat ${MCONF} | jq -r '.pretrain_model')
SAMPLE_CONFIG=$(cat ${MCONF} | jq -r '.sample_request_config')
SAVE=$(cat ${MCONF} | jq -r '.save')

# Set the default value of the getopts variable 
GPU="all"
CONTAINER_NAME="rag"
IMAGE_NAME="${USER}/${BASE_NAME}:${TAG_VER}"

CURRENT_DIR=$(pwd)

DOCKER_CMD="docker run \
--gpus ${GPU} \
-it \
--ipc=host \
--network host \
-v ${CURRENT_DIR}:/workspace/ \
--rm \
--name ${CONTAINER_NAME} \
${IMAGE_NAME} bash"


# ---------------------------------------------------------
echo -e "${YELLOW}"
echo "----- Command: ${DOCKER_CMD} -----"
echo -e "${NC}"

bash -c "${DOCKER_CMD}"

# ---------------------------------------------------------

echo -e "${YELLOW}"
echo "----- Close container -----"
echo -e "${NC}"
docker stop ${CONTAINER_NAME}
