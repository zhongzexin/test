#!/bin/bash
cd `dirname $0`
BIN_DIR=`pwd`
cd ..
PYTHONPATH=`pwd`
export PYTHONPATH
echo $PYTHONPATH


echo -e "Starting app ...\c"
cd script
exec python3  run.py