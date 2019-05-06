#!/bin/sh

docker stop dgl-container
docker rm dgl-container
docker build . -t dgl:v1
# non-interactive
docker run --name dgl-container dgl:v1 /bin/bash /root/run_examples.sh 

# attach if necessary: docker  exec -it dgl-container /bin/bash
