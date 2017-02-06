// fastivectorbin/fast-ivector-diag-randsvd-est.cc

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
      "Obtain Maximum Likelihood estimate of the parameters of a diagonal\n" 
      "covariance fast ivector model from accumulated stats\n"
      "The algorithm is based on randomized SVD. For details, refer to : \n"
      "\'\'Finding structure with randomness: Probabilistic algorithms for\n" 
      "constructing approximate matrix decompositions, N Halko et al, SIAM 2011\n"
      "Usage:  fast-ivector-diag-randsvd-est [options] <B1> ... <Bn> <Q>\n"
      "        <stats-NFS> <ivec-mdl-wxfilename> \n"
      "e.g.: fast-ivector-diag-randsvd-est B.1.mat B.2.mat Q.mat\n" 
      "      stats_NFS.global fastivec.mdl\n";

    ParseOptions po(usage);
    FastIvectorEstimationOptions est_opts;
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    est_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_outfilename = po.GetArg(po.NumArgs()), 
                stats_NFS_rxfilename = po.GetArg(po.NumArgs()-1),
                Q_rxfilename = po.GetArg(po.NumArgs()-2);

    int32 num_matrices = po.NumArgs() - 3;
    Timer time;
    // Read and combine B matrices
    std::vector<Matrix<BaseFloat> > B_vec(num_matrices);
    int n = 0;
    for(int32 i = 0; i < num_matrices; i++) {
      ReadKaldiObject(po.GetArg(i+1),&(B_vec[i]));
      n = n + B_vec[i].NumCols();
    }
    int32 num_rows_b = B_vec[0].NumRows();
    Matrix<BaseFloat> B(num_rows_b,n);
    int32 count = 0;
    for(int32 i = 0; i < num_matrices; i++) {
      B.Range(0,B.NumRows(),count,B_vec[i].NumCols()).CopyFromMat(B_vec[i]);
      count = count + B_vec[i].NumCols();
    }  

    // Read the zeroth and second order stats
    FastIvectorDiagStats stats_NFS;
    stats_NFS.Read(stats_NFS_rxfilename);
    int32 num_gauss = stats_NFS.NumGauss(), feat_dim = stats_NFS.FeatDim();
    std::vector<Vector<BaseFloat> > S(num_gauss);
    for(int32 c = 0; c < num_gauss; c++) {
      stats_NFS.GetS(&S[c],c);
    }
    Vector<BaseFloat> N;
    stats_NFS.GetN(&N);
    int64 num_utt = stats_NFS.GetNumUtt();
    BaseFloat avg_utt_dur =  N.Sum()/num_utt;

    // Read the Q Matrix
    Matrix<BaseFloat> Q;
    ReadKaldiObject(Q_rxfilename,&Q);

    // Make sure all the dimensions are consistent with provided options
    int32 k = est_opts.ivec_dim, p = est_opts.p, m = feat_dim*num_gauss;
    KALDI_ASSERT(B.NumRows() == k + p);
    KALDI_ASSERT((Q.NumRows() == k+p) && (Q.NumCols() == m));

    // 5. Compute the SVD of B : B = Ut * D * V'
    BaseFloat t = time.Elapsed();
    Matrix<BaseFloat> U_t(k+p,k+p); Vector<BaseFloat> D(k+p);
    B.Svd(&D,&U_t,NULL);
    SortSvd(&D,&U_t);
    KALDI_LOG << "For " << num_utt << " utterances, singular Values are : ";
    KALDI_LOG << D;

    // 6. Obtain U = Q * Ut
    Matrix<BaseFloat> U(m,k,kSetZero);
    U.AddMatMat(1.0,Q,kTrans,U_t.Range(0,k+p,0,k),kNoTrans,1.0);
    KALDI_LOG << "Smaller SVD obtained in " << time.Elapsed() - t << " s";

    // Transform U to get S^-1 * T
    Vector<BaseFloat> D_T(k);
    for(int32 i = 0; i < k; i++) {
      D_T(i) = (D(i)/avg_utt_dur) * (D(i)/num_utt) - 2.0/avg_utt_dur;
      if (D_T(i) > 0) {
        D_T(i) = sqrt(D_T(i));
      } else {
        KALDI_WARN << "Unable to estimate " << k << " dimensional subspace." 
                   << "Returning subspace of dimension " << i << " instead.";
        k = i;
        D_T = D_T.Range(0,k);
        U = U.Range(0,m,0,k);
        break;
      }
    }
    t = time.Elapsed();
    for(int32 i = 0; i < num_gauss*feat_dim; i++) {
      int32 c_i = i/feat_dim, dim_i = (i % feat_dim);
      U.Row(i).Scale(1.0/pow(S[c_i](dim_i),0.5));
    }
    U.MulColsVec(D_T);
    Matrix<BaseFloat> &S_Inv_T(U);
    // Create and write the ivector model
    Vector<BaseFloat> F_mean; stats_NFS.GetF(F_mean);
    Matrix<BaseFloat> M(num_gauss,feat_dim); M.CopyRowsFromVec(F_mean);
    FastIvectorDiag FastIvec_Model(M,S,S_Inv_T,D_T);
    FastIvec_Model.Write(model_outfilename,binary);    
    KALDI_LOG << "Final model obtained after SVD in " << time.Elapsed() - t << " s";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
