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
stage=-1
num_gselect=30 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.0001 # Minimum posterior to use (posteriors below this are pruned out)
ivec_dim=400 # Subspace dimensionality
p=100 # Oversampling parameter for randomized SVD
vad_scp=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <ubm-mdl> <data-dir> <stats-dir>"
  echo " e.g.: $0 exp/ivec_mdl/ubm/final.dubm data/train exp/fast_ivec_mdl/stats"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|4>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|-2>                               # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  exit 1;
fi

ubm_mdl=$1
data=$2
stats_dir=$3

[ ! -f $data/feats.scp ] && echo "No such file $data/feats.scp" && exit 1;

# Split the data dir for parallelization
mkdir -p $stats_dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

# Set up features
ubm_dir=`dirname $ubm_mdl`
delta_opts=`cat $ubm_dir/delta_opts`
if [ -z "$vad_scp" ]; then
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
else
  feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp:$vad_scp ark:- |"
fi

# Do Gaussian selection and posterior extraction
if [ $stage -le -1 ]; then
  echo $nj > $stats_dir/num_jobs
  echo "$0: doing Gaussian selection and posterior computation"
  $cmd JOB=1:$nj $stats_dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $ubm_mdl "$feats" ark:- \| \
    gmm-global-gselect-to-post --min-post=$min_post $ubm_mdl "$feats" \
       ark,s,cs:- "ark:|gzip -c >$stats_dir/post.JOB.gz" || exit 1;
else
  if ! [ $nj -eq $(cat $stats_dir/num_jobs) ]; then
    echo "Num-jobs mismatch $nj versus $(cat $stats_dir/num_jobs)"
    exit 1
  fi
fi

# Extract the stats
echo "$0: Extracting stats for ivector model estimation"
post_dim=$(gmm-global-info $ubm_mdl | grep number | cut -f 4 -d " ")
$cmd JOB=1:$nj $stats_dir/log/acc_stats.JOB.log \
  fast-ivector-diag-acc-stats \
    --p=$p \
    --ivec-dim=$ivec_dim \
    "$feats" "ark,o:gunzip -c $stats_dir/post.JOB.gz|" $post_dim \
    ark:$stats_dir/stats_N.JOB ark:$stats_dir/stats_F.JOB \
    $stats_dir/stats_NFS.JOB || touch $stats_dir/.error &
wait
[ -f $stats_dir/.error ] && echo "Error accumulating stats" && rm $stats_dir/.error && exit 1;
echo "Succeeded in obtaining stats"
