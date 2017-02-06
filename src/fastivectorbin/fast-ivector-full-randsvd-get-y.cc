// fastivectorbin/fast-ivector-full-randsvd-get-y.cc

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
      "Get Y = F*O , where O is Gaussian Random matrix, for randomized SVD.\n"
      "For details, refer to :\n"
      "\'\'Finding structure with randomness: Probabilistic algorithms for\n"
      "constructing approximate matrix decompositions\'\', N Halko et al, SIAM 2011\n"
      "Usage:  fast-ivector-full-randsvd-get-y [options] <stats-NFS>\n"
      "        <stats-N-rspecifier> <stats-F-rspecifier> <mat-wxfilename> \n"
      "e.g.: fast-ivector-full-randsvd-get-y  stats_NFS ark:stats_N.1\n"
      "                                       ark:stats_F.1 Y.1\n";

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
    std::string stats_NFS_rxfilename = po.GetArg(1),
                stats_N_rspecifier = po.GetArg(2),
                stats_F_rspecifier = po.GetArg(3),
                mat_wxfilename = po.GetArg(4);
    Timer time;

    // Read the zeroth and second order stats
    FastIvectorFullStats stats_NFS;
    stats_NFS.Read(stats_NFS_rxfilename);

    int32 num_gauss = stats_NFS.NumGauss(), feat_dim = stats_NFS.FeatDim();

    // Get the normalization matrices
    Vector<BaseFloat> F_mean; stats_NFS.GetF(F_mean);
    std::vector<TpMatrix<BaseFloat> > invsqrt_S(num_gauss);
    for(int32 c = 0; c < num_gauss; c++) {
      SpMatrix<BaseFloat> inv_Sc;
      stats_NFS.GetS(&inv_Sc,c);
      inv_Sc.Invert();
      invsqrt_S[c].Resize(feat_dim,kSetZero);
      invsqrt_S[c].Cholesky(inv_Sc);
    }

 
    // Stage A
    // 1. Multiply with a Gaussian Random Matrix : Get Y <- F * O 
    int32 k = est_opts.ivec_dim, p = est_opts.p, batch_size = est_opts.batch_size;
    int32 m = num_gauss*feat_dim;
    Matrix<BaseFloat> Y(m,k + p,kSetZero);
    BaseFloat t = time.Elapsed();
    SequentialBaseFloatVectorReader stats_F_reader(stats_F_rspecifier);
    RandomAccessBaseFloatVectorReader stats_N_reader(stats_N_rspecifier);
    Matrix<BaseFloat> N_batch(batch_size,num_gauss),
                      F_batch(batch_size,m);
    int n = 0;
    while(! stats_F_reader.Done()) {
      int32 j;
      for(j = 0; j < batch_size; j++) {
        if(stats_F_reader.Done()) break;
        std::string utt = stats_F_reader.Key();
        SubVector<BaseFloat> F(F_batch.Row(j)),
                             N(N_batch.Row(j));
        F.CopyFromVec(stats_F_reader.Value());
        N.CopyFromVec(stats_N_reader.Value(utt));
        N.ApplyFloor(1e-3); N.ApplyPow(0.5);
        stats_F_reader.Next();
        n++;
      }
      // Take the submatrix for the last batch
      SubMatrix<BaseFloat> F_submat(F_batch,0,j,0,m),
                           N_submat(N_batch,0,j,0,num_gauss);
      F_submat.AddVecToRows(-1.0,F_mean);
      F_submat.MulRowsGroupMat(N_submat);
      Matrix<BaseFloat> F_normalized(m,j,kSetZero);
      // Normalize the stats
      for(int32 c = 0; c < num_gauss; c++) {
        SubMatrix<BaseFloat> Fc(F_submat,0,j,c*feat_dim,feat_dim);
        SubMatrix<BaseFloat> Fc_normalized(F_normalized,c*feat_dim,feat_dim,0,j);
        Fc_normalized.AddTpMat(1.0,invsqrt_S[c],kTrans,Fc,kTrans,1.0);
      }
      Matrix<BaseFloat> O(j,k + p);
      O.SetRandn();
      Y.AddMatMat(1.0,F_normalized,kNoTrans,O,kNoTrans,1.0);
    }
    WriteKaldiObject(Y,mat_wxfilename,binary);
    KALDI_LOG << "Obtained Y in " << time.Elapsed() - t << " s";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
