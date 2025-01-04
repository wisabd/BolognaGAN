#!/bin/bash
#Enable debugging commands
set -x

#Disable core dumps
ulimit -c 0

#Add CVMS libraries, resources
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-e19-gcc11-opt/setup.sh

#Set up directories
HOME="/afs/cern.ch/user/a/abdullam/"
APPTAINER_FILE="/eos/user/a/abdullam/BolotestBranch/bologantainer/BoloGANtainer.sif"
DATADIR ="/eos/user/a/abdullam/Data_CaloGAN/data"
CODEDIR ="afs/cern.ch/user/a/abdullam/BoloGAN"
BINNING_FILE=binning.xml

#CUDA options
#To enable CUDA:
export APPTAINERENV_CUDA_VISIBLE_DEVICES="0"
#Use this line to disable CUDA
#export APPTAINERENV_CUDA_VISIBLE_DEVICES=""

#Training Options to be passed to train.py later
PARTICLE="pions"
FIRST_EPOCH=0
EPOCHS=1000
SAMPLES_RANGE="All"
ETA_MIN=20
ETA_MAX=25
#ENERGY_RANGE=
STEPS_PER_EXECUTION=3200

for steps in 1 100 200 400 800 1600 3200; do
  export STEPS_PER_EXECUTION=$steps
  NAME="tesi/matrix_steps_per_execution_$STEPS_PER_EXECUTION"
  WORKINGDIR="/eos/user/a/abdullam/Pionsrun/$NAME/$PARTICLE/$SAMPLES_RANGE/${ETA_MIN}_${ETA_MAX}"

  #Create working directory
  mkdir -p "$WORKINGDIR"

  #Miscellaneous Options
  export USE_PARQUET=FALSE    #Set to TRUE if loading data from parquet files
  export PYTHONCACHEPREFIX="/tmp/$USER__pycache__"

  #Change to code directory
  cd "$CODEDIR" || exit 1 #Exit if cd fails

  python3 train.py \
      --output_dir_gan  "${WORKINGDIR}" \
      --input_dir       "${DATADIR}"  \
      --particle        "${PARTICLE}"  \
      --eta_min         "${ETA_MIN}"  \
      --eta_max         "${ETA_MAX}"  \
      --firstEpoch      "${FIRST_EPOCH}" \
      --energy_range    "${SAMPLES_RANGE}"  \
      --epochs          "${EPOCHS}"  \
      --binning         "${BINNING_FILE}"  \
      --sample-interval "${SAMPLE_INTERVAL}"  \
      --steps-per-execution "${STEPS_PER_EXECUTION}"

  sleep 120

done
  
