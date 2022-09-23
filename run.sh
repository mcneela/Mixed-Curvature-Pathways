#!/usr/bin/env bash

# code file should default to code.tar.gz if you're using submit.sh and submit.py
CODE_FN=code.tar.gz
# if your environment file is called something different (like my_env.tar.gz), set it here...
ENV_FN=env.tar.gz
# same with your data file
DATA_FN=data.tar.gz


# exit if any command fails...
set -e

# create output directory for condor logs
mkdir -p output/condor_logs

# echo some HTCondor job information
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "System: $(uname -spo)"
echo "_CONDOR_JOB_IWD: $_CONDOR_JOB_IWD"
echo "Cluster: $CLUSTER"
echo "Process: $PROCESS"
echo "RunningOn: $RUNNINGON"

# this makes it easier to set up the environments, since the PWD we are running in is not $HOME
export HOME=$PWD

# un-tar the code repo (transferred from submit node), removing the enclosing folder with strip-components
if [ -f "$CODE_FN" ]; then
  echo "Extracting $CODE_FN"
  tar -xf code.tar.gz --strip-components=1
  rm code.tar.gz
fi

# set up python environment
if [ "$(ls 2>/dev/null -Ubad1 -- $ENV_FN.* | wc -l)" -gt 0 ];
then
  echo "Combining split environment files"
  cat $ENV_FN.* > $ENV_FN
  chmod 644 $ENV_FN
  rm $ENV_FN.*
fi

# un-tar the environment files
if [ -f "$ENV_FN" ]; then
  echo "Extracting $ENV_FN"
  mkdir env
  tar -xzf $ENV_FN -C env
  rm $ENV_FN
fi

# it seems the python environment needs to be reactivated on checkpoint
# would be great if there was a way to check if the environment is already active so i could wrap in if statement
echo "Activating Python environment"
export PATH
. env/bin/activate

echo "Launching train.py"
python3 code/train.py @"args/${PROCESS}.txt" --cluster="$CLUSTER" --process="$PROCESS" --github_tag="$GITHUB_TAG"

