// fastivectorbin/fast-ivector-full-randsvd-get-b.cc

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
      "Get B = Q'*F for randomized SVD. For details, refer to : \n"
      "\'\'Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions\'\'\n"
      "N Halko et al, SIAM 2011\n"
      "Usage:  fast-ivector-full-randsvd-get-b [options] <NS-stats-rxfilename> <F-stats-rspecifier> <mat-rxfilename> <mat-wxfilename> \n"
      "e.g.: fast-ivector-full-randsvd-get-b  NS.stats ark:stats_F.1 Y.mat B.1.mat \n";

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

    // Read Q
    Matrix<BaseFloat> Q;
    ReadKaldiObject(mat_rxfilename,&Q);

    // Get number of utterances in archive, required for resizing the output matrix appropriately
    int32 n = 0;
    BaseFloat t = time.Elapsed();
    SequentialBaseFloatVectorReader stats_reader_for_count(stats_F_rspecifier);
    for(;! stats_reader_for_count.Done();stats_reader_for_count.Next()) n++;
    stats_reader_for_count.Close();
    KALDI_LOG << "Archive length is " << n << ", obtained in " << time.Elapsed() - t << " s";

    // Stage B
    // 4. Get B <- Q' * F
    t = time.Elapsed();
    int32 batch_size = est_opts.batch_size, m = num_gauss*feat_dim, num_rows_q = Q.NumRows();
    Matrix<BaseFloat> B(num_rows_q,n,kSetZero);
    int32 count = 0;
    SequentialBaseFloatVectorReader stats_reader(stats_F_rspecifier);
    Matrix<BaseFloat> F_batch(batch_size,m);
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
      for(int32 c = 0; c < num_gauss; c++) {
        SubMatrix<BaseFloat> Fc(F_submat,0,j,c*feat_dim,feat_dim);
        SubMatrix<BaseFloat> Fc_normalized(F_normalized,c*feat_dim,feat_dim,0,j);
        Fc_normalized.AddTpMat(1.0,invsqrt_S[c],kTrans,Fc,kTrans,1.0);
      }
      SubMatrix<BaseFloat> B_submat(B,0,B.NumRows(),count,j);
      B_submat.AddMatMat(1.0,Q,kNoTrans,F_normalized,kNoTrans,1.0);
      count = count + j;
    }
    WriteKaldiObject(B,mat_wxfilename,binary);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
