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
nj=16
cmd="run.pl"
stage=-2
num_gselect=30 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.0001 # Minimum posterior to use (posteriors below this are pruned out)
ivec_dim=-1
vad_scp=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <data> <ivec-mdl-dir> <ivec-out-dir>"
  echo " e.g.: $0 data/train exp/fast_ivec_mdl/ivec.mdl exp/fast_ivectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|4>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|-2>                               # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  exit 1;
fi

data=$1
ivec_mdl_dir=$2
ivec_out_dir=$3

for f in $data/feats.scp $ivec_mdl_dir/ubm/final.ubm; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done
mkdir -p $ivec_out_dir/log

# Split the data dir for parallelization
sdata=$data/split$nj;
utils/split_data.sh --per-utt $data $nj || exit 1;

# Set up features
delta_opts=`cat $ivec_mdl_dir/ubm/delta_opts`
if [ -z "$vad_scp" ]; then
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
else
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp:$vad_scp ark:- |"
fi

# Do Gaussian selection and posterior extraction
if [ $stage -le -2 ]; then
  $cmd $ivec_out_dir/log/convert.log \
    fgmm-global-to-gmm $ivec_mdl_dir/ubm/final.ubm $ivec_mdl_dir/ubm/full_to_diag.dubm || exit 1;
fi
if [ $stage -le -1 ]; then
  echo $nj > $ivec_out_dir/num_jobs
  echo "$0: Doing Gaussian selection and posterior computation"
  $cmd JOB=1:$nj $ivec_out_dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $ivec_mdl_dir/ubm/full_to_diag.dubm "$feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $ivec_mdl_dir/ubm/final.ubm "$feats" \
       ark,s,cs:- "ark:|gzip -c >$ivec_out_dir/post.JOB.gz" || exit 1;
else
  if ! [ $nj -eq $(cat $ivec_out_dir/num_jobs) ]; then
    echo "Num-jobs mismatch $nj versus $(cat $ivec_out_dir/num_jobs)"
    exit 1
  fi
fi

# Extract ivectors
if [ $stage -le 0 ]; then
  echo "$0: Extracting ivectors"
  $cmd JOB=1:$nj $ivec_out_dir/log/fastivec_extract.JOB.log \
    fast-ivector-full-extract --ivec-dim="$ivec_dim" $ivec_mdl_dir/ivec.mdl "$feats" "ark:gunzip -c $ivec_out_dir/post.JOB.gz|" ark,scp:$ivec_out_dir/ivectors.JOB.ark,$ivec_out_dir/ivectors.JOB.scp || touch $ivec_out_dir/.error & 
  wait
  [ -f $ivec_out_dir/.error ] && echo "Error extracting ivectors" && rm $ivec_out_dir/.error && exit 1;
fi

# Combine ivectors across jobs
echo "$0: Combining ivectors across jobs"
for j in $(seq $nj); do cat $ivec_out_dir/ivectors.$j.scp; done >$ivec_out_dir/ivectors.scp || exit 1;
echo "$0: Succeeded in extracting ivectors"
