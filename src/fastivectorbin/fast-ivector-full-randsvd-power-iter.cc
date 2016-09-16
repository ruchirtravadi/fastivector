// fastivectorbin/fast-ivector-full-randsvd-power-iter.cc

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
      "Power iteration : Get Y <- FF'*Y for randomized SVD. For details, refer to : \n"
      "\'\'Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions\'\'\n"
      "N Halko et al, SIAM 2011\n"
      "Usage:  fast-ivector-full-randsvd-power-iter [options] <NS-stats-rxfilename> <F-stats-rspecifier> <mat-rxfilename> <mat-wxfilename>\n"
      "e.g.: fast-ivector-full-randsvd-power-iter  stats_NS ark:stats_F.1 Y.powiter0.mat Y.powiter1.1.mat \n";

    ParseOptions po(usage);
    FastIvectorEstimationOptions est_opts;
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    est_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string stats_NS_rxfilename = po.GetArg(1),
                stats_F_rspecifier = po.GetArg(2),
                mat_rxfilename = po.GetArg(3),
                mat_wxfilename = po.GetArg(4);
    Timer time;

    // Read the zeroth and second order stats
    FastIvectorFullStats stats_NS;
    stats_NS.Read(stats_NS_rxfilename);

    int32 num_gauss = stats_NS.NumGauss(), feat_dim = stats_NS.FeatDim();

    // Get the normalization matrices
    std::vector<TpMatrix<BaseFloat> > invsqrt_S(num_gauss);
    for(int32 c = 0; c < num_gauss; c++) {
      SpMatrix<BaseFloat> inv_Sc;
      stats_NS.GetS(&inv_Sc,c);
      inv_Sc.Invert();
      invsqrt_S[c].Resize(feat_dim,kSetZero);
      invsqrt_S[c].Cholesky(inv_Sc);
    }
 
    // Read Y 
    Matrix<BaseFloat> Y;
    ReadKaldiObject(mat_rxfilename,&Y);

    // Stage A
    // 2. Power iteration : Get Y <- (F*F') * Y
    BaseFloat t = time.Elapsed();
    int32 batch_size = est_opts.batch_size, p = est_opts.p, k = est_opts.ivec_dim, m = num_gauss*feat_dim;
    KALDI_ASSERT(k + p == Y.NumCols());
    Matrix<BaseFloat> Y_out(m,k+p,kSetZero);
    SequentialBaseFloatVectorReader stats_reader(stats_F_rspecifier);
    Matrix<BaseFloat> F_batch(batch_size,m);
    int32 count = 0;
    while(! stats_reader.Done()) {
      int32 j;
      for(j = 0; j < batch_size; j++) {
        if(stats_reader.Done()) break;
        std::string utt = stats_reader.Key();
        SubVector<BaseFloat> F(F_batch.Row(j));
        F.CopyFromVec(stats_reader.Value());
        stats_reader.Next();
      }
      // Take the submatrix for the last batch
      SubMatrix<BaseFloat> F_submat(F_batch,0,j,0,m);
      Matrix<BaseFloat> F_normalized(m,j,kSetZero);
      // Normalize the stats
      for(int32 c = 0; c < num_gauss; c++) {
        SubMatrix<BaseFloat> Fc(F_submat,0,j,c*feat_dim,feat_dim);
        SubMatrix<BaseFloat> Fc_normalized(F_normalized,c*feat_dim,feat_dim,0,j);
        Fc_normalized.AddTpMat(1.0,invsqrt_S[c],kTrans,Fc,kTrans,1.0);
      }
      Matrix<BaseFloat> Y_temp(j,k+p,kSetZero);
      // Y_temp = F' * Y
      Y_temp.AddMatMat(1.0,F_normalized,kTrans,Y,kNoTrans,1.0);
      // Y_out = Y_out + F * Y_temp
      Y_out.AddMatMat(1.0,F_normalized,kNoTrans,Y_temp,kNoTrans,1.0);
      count = count + j;
    }  
    KALDI_LOG << "Finished power iteration in " << time.Elapsed() - t << " s";
    WriteKaldiObject(Y_out,mat_wxfilename,binary);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
