#!/bin/bash

docker build --no-cache -t er_dl .
docker container kill er_dl
docker container rm er_dl
docker container run -it -p 80:80 --name er_dl er_dl
