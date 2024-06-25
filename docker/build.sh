#!/bin/bash
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# --------------------------------------------------------
CONF="./docker/configuration/version.json"
ROOT=$(dirname "${FILE}")

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
#   echo -e "${RED}"
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
  echo -e "${YELLOW}"
  echo '   
   _____     _ _   _ _         
  | __  |_ _| |_|_| |_|___ ___ 
  | __ -| | | | | . | |   | . |
  |_____|___|_|_|___|_|_|_|_  |
                          |___|

  '
  echo -e "${NC}"
}

# ---------------------------------------------------------
# help
function help(){
	echo "-----------------------------------------------------------------------"
	echo "Build the RAG environment."
	echo
	echo "Syntax: scriptTemplate [-m|h]"
	echo "options:"
    echo "m		Print information with magic"
	echo "h		help."
	echo "-----------------------------------------------------------------------"
}

while getopts "mh" option; do
	case $option in
        m )
			magic=true
			;;
		h )
			help
			exit
			;;
		\? )
			help
			exit
			;;
		* )
			help
			exit
			;;
	esac
done

init_word

# Install jq
echo -e "${YELLOW}"
echo "----- Installing JQ -----"
echo -e "${NC}"

if ! type jq >/dev/null 2>&1; then
    sudo apt-get install -y jq
else
    echo 'The jq has been installed.';
fi

# ---------------------------------------------------------
# Install pyinstaller
echo -e "${YELLOW}"
echo "----- Installing pyinstaller -----"
echo -e "${NC}"
pip install pyinstaller

# --------------------------------------------------------
# Parse information from configuration
USER=$(cat ${CONF} | jq -r '.USER')
BASE_NAME=$(cat ${CONF} | jq -r '.PROJECT')
TAG_VER=$(cat ${CONF} | jq -r '.VERSION')

# --------------------------------------------------------
# Concate name
IMAGE_NAME="${USER}/${BASE_NAME}:${TAG_VER}"
echo -e "${YELLOW}"
echo "----- Concatenate docker image name: ${IMAGE_NAME} -----"
echo -e "${NC}"

# --------------------------------------------------------
# Build main docker image
echo -e "${YELLOW}"
echo " +-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+"
echo " |B|u|i|l|d| |m|a|i|n| |d|o|c|k|e|r| |i|m|a|g|e|"
echo " +-+-+-+-+-+ +-+-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+"
echo -e "${NC}"

docker build -t "${IMAGE_NAME}" \
-f "${ROOT}/docker/Dockerfile" . 



# ---------------------------------------------------------
# Push dockerhub
# IMAGE_NAME="${USER}/${BASE_NAME}:${TAG_VER}"
# echo -e "${GREEN}"
# echo "----- Push dockerhub ${IMAGE_NAME} -----"
# echo -e "${NC}"
# docker push ${IMAGE_NAME}