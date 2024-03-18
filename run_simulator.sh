#!/bin/bash

export GCC_VERSION=7.3.0
export CUDA_VERSION=10.1

export ACCELSIM_ROOT_DIR=$(pwd)/..
if CUDA_VERSION=10.1; then
  export GPGPUSIM_LIB_PATH=$ACCELSIM_ROOT_DIR/gpgpu-sim/lib/gcc-${GCC_VERSION}/cuda-10010/release
fi
if [ -z "$GPGPUSIM_LIB_PATH" ]; then
  echo "Error: GPGPUSIM_LIB_PATH is not set"
  exit 1
fi

#TARGET_MEM can be HMS, HMS-BP, HMS-BP-CTC, HBM, PCM
export TARGET_MEM=HMS
export TARGET_FOOTPRINT_RATIO=75 # 75% of the workload's memory footprint can be in HBM (R_HBM = 75%)
export TARGET_TRACE_DIR=$ACCELSIM_ROOT_DIR/gpgpu-sim/example_traces/color_maxmin_example/traces

# call make_run_environment.py to generate run environment. printed output is the output directory path.
# catch the output directory path and use it to run the simulator
python $ACCELSIM_ROOT_DIR/gpgpu-sim/make_run_environment.py \
      -m $TARGET_MEM\
      -t $TARGET_TRACE_DIR\
      -f $TARGET_FOOTPRINT_RATIO\
      -o $ACCELSIM_ROOT_DIR/gpgpu-sim\
      -c $ACCELSIM_ROOT_DIR/gpgpu-sim/configs/$TARGET_MEM

# check run_environment.txt file exists
if [ ! -f run_environment.txt ]; then
  echo "Error: run_environment.txt does not exist"
  exit 1
fi
export RUN_ENVIRONMENT=$(cat run_environment.txt)
rm run_environment.txt

cd $RUN_ENVIRONMENT

# Run the simulator
export LD_LIBRARY_PATH=$GPGPUSIM_LIB_PATH:$LD_LIBRARY_PATH

$ACCELSIM_ROOT_DIR/bin/release/accel-sim.out \
 -config ./gpgpusim.config \
 -config ./trace.config \
 -trace $TARGET_TRACE_DIR/kernelslist.g \
 > output/gpgpusim.result