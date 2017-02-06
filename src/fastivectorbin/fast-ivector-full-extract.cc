// fastivectorbin/fast-ivector-full-extract.cc

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
#include "base/timer.h"
#include "fastivector/fast-ivector.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
      const char *usage =
      "Extract ivectors from the full covariance fast ivector model\n"
      "Usage:  fast-ivector-full-extract [options] <ivec-mdl> <feats-rspecifier>\n"
      "                                      <post-rspecifier> <ivec-wspecifier>\n"
      "e.g.: fast-ivector-full-extract fast-ivec.mdl scp:feats.scp ark:post.ark ark:ivectors.ark\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    int32 ivec_dim = -1;
    po.Register("ivec-dim", &ivec_dim, "Truncate ivector to specified dimension, default (= -1) is no truncation");

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    // Read the specified inputs
    std::string ivecmdl_rxfilename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        ivec_wspecifier = po.GetArg(4);

    FastIvectorFull ivec_mdl;
    ivec_mdl.Read(ivecmdl_rxfilename);

    if (ivec_dim == -1) ivec_dim = ivec_mdl.IvecDim();
    if (ivec_dim > ivec_mdl.IvecDim()) {
      KALDI_WARN << "Specified ivector dimension " << ivec_dim << " larger than allowed by the model : " << ivec_mdl.IvecDim();
      KALDI_WARN << "The ivectors generated will be of length " << ivec_mdl.IvecDim();
      ivec_dim = ivec_mdl.IvecDim();
    }

    // Loop across all utterance to calculate stats
    SequentialBaseFloatMatrixReader feature_reader(feats_rspecifier);
    RandomAccessPosteriorReader post_reader(post_rspecifier);
    BaseFloatVectorWriter ivec_writer(ivec_wspecifier);

    int32 num_utt = 0;
    Timer time;

    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      num_utt++;
      std::string utt = feature_reader.Key();
      Matrix<BaseFloat> feats = feature_reader.Value();
      Posterior post = post_reader.Value(utt);
      Vector<BaseFloat>  x_utt(ivec_dim);
      ivec_mdl.GetIvector(feats,post,x_utt);
      ivec_writer.Write(utt,x_utt);
    }
    KALDI_LOG << "Finished in " << time.Elapsed() << " s ";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
