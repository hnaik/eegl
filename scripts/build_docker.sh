#!/bin/bash

set -euo pipefail

name=$1

root_dir=$(dirname $(dirname $(realpath $0)))
echo $root_dir

docker build . -t hnaik/egr-$name:latest -f $root_dir/cr/$name/Dockerfile

rev=$(git rev-parse --short=4 HEAD)
docker image tag $name:latest hnaik/egr-$name:$rev

docker push hnaik/egr-$name:$rev
docker push hnaik/egr-$name:latest
