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

# Begin configuration section.
cmd="run.pl"
stage=0
cleanup=true
binary=false
est_opts=""
q=1
ivec_dim=400
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <stats-dir> <ivec-mdl-dir>"
  echo " e.g.: $0 exp/fast_ivec_mdl/stats exp/fast_ivec_mdl"
  exit 1;
fi

stats_dir=$1
out_dir=$2

# Make sure all the necessary files exist
nj=$(cat $stats_dir/num_jobs)
for i in `seq 1 $nj`; do
  [ ! -f $stats_dir/stats_N.${i} ] && echo "No such file $stats_dir/stats_N.${i}" && exit 1;
  [ ! -f $stats_dir/stats_F.${i} ] && echo "No such file $stats_dir/stats_F.${i}" && exit 1;
  [ ! -f $stats_dir/stats_NFS.${i} ] && echo "No such file $stats_dir/stats_NFS.${i}" && exit 1;
done
mkdir -p $out_dir/log

echo "Started estimation, will parallelize over $nj jobs"

if [ $stage -le 1 ]; then
  start=$SECONDS
  # Get the combined zeroth and second order stats
  fast-ivector-full-combine-stats $stats_dir/stats_NFS.* $stats_dir/stats_NFS.global
  echo "Obtained the weights and covariances in $(( SECONDS - start )) sec"
fi
 
# Get the Y matrix
if [ $stage -le 2 ]; then
  start=$SECONDS
  $cmd JOB=1:$nj $out_dir/log/est_y.JOB.log \
    fast-ivector-full-randsvd-get-y --ivec-dim=$ivec_dim $stats_dir/stats_NFS.global \
      ark:$stats_dir/stats_N.JOB ark:$stats_dir/stats_F.JOB \
      $out_dir/Y.poweriter0.JOB.mat
  matrix-sum $out_dir/Y.poweriter0.*.mat $out_dir/Y.poweriter0.mat
  echo "Obtained the Y matrix in  $(( SECONDS - start )) sec"
  if $cleanup; then
    rm $out_dir/Y.poweriter0.*.mat
  fi
fi

# Power Iteration
if [ $stage -le 3 ]; then
  start=$SECONDS
  for pow_it in `seq 1 ${q}`; do
    start=$SECONDS
    $cmd JOB=1:$nj $out_dir/log/pow_iter${q}.JOB.log \
      fast-ivector-full-randsvd-power-iter --ivec-dim=$ivec_dim \
        $stats_dir/stats_NFS.global \
        ark:$stats_dir/stats_N.JOB ark:$stats_dir/stats_F.JOB \
        $out_dir/Y.poweriter$((q-1)).mat $out_dir/Y.poweriter${q}.JOB.mat
    matrix-sum $out_dir/Y.poweriter${q}.*.mat $out_dir/Y.poweriter${q}.mat
    echo "Power iteration ${q} finished in $(( SECONDS - start )) sec"
    if $cleanup; then
      if [ $q -ge 1 ]; then
        rm $out_dir/Y.poweriter$((q-1)).*
        rm $out_dir/Y.poweriter${q}.*.mat
      fi
    fi
  done
fi

# Get the Q matrix
if [ $stage -le 3 ]; then
  start=$SECONDS
  fast-ivector-full-randsvd-get-q --ivec-dim=$ivec_dim $out_dir/Y.poweriter${q}.mat $out_dir/Q.mat
  echo "Obtained the Q matrix in  $(( SECONDS - start )) sec"
  if $cleanup; then
    rm $out_dir/Y.poweriter${q}.mat
  fi
fi

# Get the B matrices
if [ $stage -le 4 ]; then
  start=$SECONDS
  $cmd JOB=1:$nj $out_dir/log/est_b.JOB.log \
    fast-ivector-full-randsvd-get-b --ivec-dim=$ivec_dim $stats_dir/stats_NFS.global \
      ark:$stats_dir/stats_N.JOB ark:$stats_dir/stats_F.JOB \
      $out_dir/Q.mat $out_dir/B.JOB.mat
  echo "Obtained the B matrices in  $(( SECONDS - start )) sec"
fi

# Get the final model by SVD
if [ $stage -le 5 ]; then
  fast-ivector-full-randsvd-est --binary=$binary --ivec-dim=$ivec_dim $out_dir/B.* \
    $out_dir/Q.mat $stats_dir/stats_NFS.global $out_dir/ivec.mdl
fi

if $cleanup; then
  rm $out_dir/B.*
  rm $out_dir/Q.mat
  rm $stats_dir/stats_NFS.global
fi
