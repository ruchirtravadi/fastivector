// fastivectorbin/fast-ivector-diag-randsvd-get-y.cc

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
      "Get Y = F*O , where O is Gaussian Random matrix, for randomized SVD. For details, refer to : \n"
      "\'\'Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions\'\'\n"
      "N Halko et al, SIAM 2011\n"
      "Usage:  fast-ivector-diag-randsvd-get-y [options] <NS-stats-infilename> <F-stats-rspecifier> <mat-wxfilename> \n"
      "e.g.: fast-ivector-diag-randsvd-get-y  stats.NS ark:stats_F.1 Y.1\n";

    ParseOptions po(usage);
    FastIvectorEstimationOptions est_opts;
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    est_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string stats_NS_rxfilename = po.GetArg(1),
                stats_F_rspecifier = po.GetArg(2),
                mat_wxfilename = po.GetArg(3);
    Timer time;

    // Read the zeroth and second order stats
    FastIvectorDiagStats stats_NS;
    stats_NS.Read(stats_NS_rxfilename);

    int32 num_gauss = stats_NS.NumGauss(), feat_dim = stats_NS.FeatDim(); 

    // Get the normalization matrices
    std::vector<Vector<BaseFloat> > invsqrt_S(num_gauss);
    for(int32 c = 0; c < num_gauss; c++) {
      Vector<BaseFloat> Sc;
      stats_NS.GetS(&Sc,c);
      invsqrt_S[c].Resize(feat_dim,kSetZero);
      invsqrt_S[c].AddVec(1.0,Sc);
      invsqrt_S[c].ApplyPow(-0.5);
    }
    // Stage A
    // 1. Multiply with a Gaussian Random Matrix : Get Y <- F * O 
    int32 k = est_opts.ivec_dim, p = est_opts.p, batch_size = est_opts.batch_size;
    int32 m = num_gauss*feat_dim;
    Matrix<BaseFloat> Y(m,k + p,kSetZero);
    BaseFloat t = time.Elapsed();
    SequentialBaseFloatVectorReader stats_reader(stats_F_rspecifier);
    Matrix<BaseFloat> F_batch(batch_size,m);
    int n = 0;
    while(! stats_reader.Done()) {
      int32 j;
      for(j = 0; j < batch_size; j++) {
        if(stats_reader.Done()) break;
        std::string utt = stats_reader.Key();
        SubVector<BaseFloat> F(F_batch.Row(j));
        F.CopyFromVec(stats_reader.Value());
        stats_reader.Next();
        n++;
      }
      // Take the submatrix for the last batch
      SubMatrix<BaseFloat> F_submat(F_batch,0,j,0,m);
      // Normalize the stats
      for(int32 c = 0; c < num_gauss; c++) {
        SubMatrix<BaseFloat> Fc(F_submat,0,j,c*feat_dim,feat_dim);
        Fc.MulColsVec(invsqrt_S[c]);
      }
      Matrix<BaseFloat> O(j,k + p);
      O.SetRandn();
      Y.AddMatMat(1.0,F_submat,kTrans,O,kNoTrans,1.0);
    }
    WriteKaldiObject(Y,mat_wxfilename,binary);
    KALDI_LOG << "Obtained Y in " << time.Elapsed() - t << " s";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
