#!/bin/bash

# Copyright 2016 Ruchir Travadi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This is the top-level training script for training a "fastivector"
# model using randomized SVD. The order of operations is:
# 1. Train UBM
# 2. Accumulate stats
# 3. Estimate Total Variability Matrix using randomized SVD

# Begin configuration section

# General configs
nj=16
cmd="run.pl"
stage=0
delta_window=3
delta_order=2
vad_scp= #Optionally supply VAD labels

# UBM training configs
num_iters_ubm=4
num_gselect_ubm=30 # Number of Gaussian-selection indices to use while training
                   # the UBM
num_frames_ubm=500000 # number of frames to keep in memory for UBM initialization
num_iters_init_ubm=20 
initial_gauss_proportion_ubm=0.5 # Start with half the target number of Gaussians
subsample_ubm=1 # subsample all features with this periodicity, in the main E-M phase.
cleanup_ubm=true
min_gaussian_weight_ubm=0.0001
remove_low_count_gaussians_ubm=true # set this to false if you need #gauss to stay fixed.
num_threads_ubm=32
parallel_opts_ubm="-pe smp 32"

# Stats accumulation configs
ivec_dim=400
p=100
num_gselect_acc_stats=30 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post_acc_stats=0.0001 # Minimum posterior to use (posteriors below this are pruned out)

# Estimation configs
cleanup_after_est=true # Clean up the intermediate files
q=1 # Number of steps for power iteration

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <data-dir> <num-gauss> <ivec-mdl-dir>"
  echo " e.g.: $0 --nj 16 data/train 512 exp/fastivec_diag"
  echo "For options, see top of script file"
  exit 1;
fi

data=$1
num_gauss=$2
outdir=$3

# Split the data dir for parallelization
sdata=$data/split$nj
utils/split_data.sh $data $nj || exit 1;

# Train UBM
if [ $stage -le 0 ]; then
  steps/fastivec_diag/train_diag_ubm.sh \
    --nj $nj \
    --cmd $cmd \
    --num-iters $num_iters_ubm \
    --num-gselect $num_gselect_ubm \
    --num-frames $num_frames_ubm \
    --num-iters-init $num_iters_ubm \
    --initial-gauss-proportion $initial_gauss_proportion_ubm \
    --subsample $subsample_ubm \
    --cleanup $cleanup_ubm \
    --min-gaussian-weight $min_gaussian_weight_ubm \
    --remove-low-count-gaussians $remove_low_count_gaussians_ubm \
    --num-threads $num_threads_ubm \
    --parallel-opts "$parallel_opts_ubm" \
    --delta-window $delta_window \
    --delta-order $delta_order \
    ${vad_scp:+ --vad-scp "$vad_scp"} \
    $data $num_gauss $outdir/ubm
fi

# Accumulate stats
if [ $stage -le 1 ]; then
  steps/fastivec_diag/acc_stats.sh \
    --nj $nj \
    --ivec-dim $ivec_dim \
    --p $p \
    --num-gselect $num_gselect_acc_stats \
    --min-post $min_post_acc_stats \
    ${vad_scp:+ --vad-scp "$vad_scp"} \
    $outdir/ubm/final.dubm $data $outdir/stats
fi
   
# Estimate ivector model from stats
if [ $stage -le 2 ]; then
  steps/fastivec_diag/est_from_stats.sh \
    --cleanup $cleanup_after_est \
    --q $q \
    --ivec-dim $ivec_dim \
    $outdir/stats $outdir
fi
