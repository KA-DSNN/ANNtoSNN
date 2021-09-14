#!/bin/bash
VERSION="latest"
HOME=.

docker build -t ann-conversion:$VERSION -f Dockerfile $HOME
docker rm $(docker ps -a | grep Exited | awk '{print $1}')
docker rmi $(docker images | grep "<none>" | awk '{print $3}')
docker run -it ann-conversion | tee $HOME/output/ann-conversion-container-`date +%y%m%d-%T`.out