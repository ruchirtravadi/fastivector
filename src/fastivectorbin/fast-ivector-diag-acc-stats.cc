// fastivectorbin/fast-ivector-diag-acc-stats.cc

// Copyright 2016 Ruchir Travadi
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "fastivector/fast-ivector.h"
#include "base/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
      const char *usage =
      "Accumulate stats for Maximum Likelihood estimation of the parameters"
      "of a diagonal covariance fast ivector model\n"
      "Usage:  fast-ivector-diag-est [options] <feats-rspecifier> <post-rspecifier>\n"
      "        <post-dim> <stats-N-wspecifier> <stats-F-wspecifier> <stats-wxfilename>\n"
      "e.g.: fast-ivector-diag-acc-stats --ivec-dim 200 scp:feats.scp ark:post.ark\n"
      "      2048 ark:stats_N.ark ark:stats_F.ark stats_NFS.global";

    ParseOptions po(usage);
    FastIvectorEstimationOptions est_opts;
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    est_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    // Read the specified inputs
    std::string feats_rspecifier = po.GetArg(1),
        post_rspecifier = po.GetArg(2),
        // third arg is an integer post_dim
        stats_N_wspecifier = po.GetArg(4),
        stats_F_wspecifier = po.GetArg(5),
        stats_outfilename = po.GetArg(6);

    // Obtain stats for all the utterances
    SequentialBaseFloatMatrixReader feature_reader(feats_rspecifier);
    RandomAccessPosteriorReader post_reader(post_rspecifier);
    Timer time;
    Matrix<BaseFloat> feats = feature_reader.Value();
    int32 num_utt = 0, feat_dim = feats.NumCols(),
          num_gauss = atoi(po.GetArg(3).c_str());

    FastIvectorDiagStats fast_ivec_diag_stats(num_gauss,feat_dim,stats_N_wspecifier,
                                              stats_F_wspecifier);

    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      num_utt++;
      std::string utt = feature_reader.Key();
      Matrix<BaseFloat> feats = feature_reader.Value();
      Posterior post = post_reader.Value(utt);
      fast_ivec_diag_stats.AccStatsForUtterance(utt,feats,post);
      KALDI_VLOG(3) << "Processed utterance " << utt << ", total " << num_utt 
                    << " utterances in " << time.Elapsed() << "s";
    }
    feature_reader.Close();

    // Write the stats
    fast_ivec_diag_stats.Write(stats_outfilename,binary);
    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
