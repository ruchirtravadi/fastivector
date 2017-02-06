// fastivectorbin/fast-ivector-full-randsvd-get-q.cc

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
      "Get Q = Orth(Y) for randomized SVD. For details, refer to : \n"
      "\'\'Finding structure with randomness: Probabilistic algorithms for\n"
      "constructing approximate matrix decompositions\'\', N Halko et al, SIAM 2011\n"
      "Usage:  fast-ivector-full-randsvd-get-q [options] <mat-rxfilename> <mat-wxfilename> \n"
      "e.g.: fast-ivector-full-randsvd-get-q  Y.mat Q.mat\n";

    ParseOptions po(usage);
    FastIvectorEstimationOptions est_opts;
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    est_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string mat_rxfilename = po.GetArg(1),
                mat_wxfilename = po.GetArg(2);
    Timer time;
    // Stage A
    // 3. G-S: Get the matrix Q whose columns form an orthonormal basis for the range of Y
    BaseFloat t = time.Elapsed();
    Matrix<BaseFloat> Y;
    ReadKaldiObject(mat_rxfilename,&Y);
    Y.Transpose();
    Y.OrthogonalizeRows();
    Matrix<BaseFloat> &Q(Y);
    KALDI_LOG << "GS finished in " << time.Elapsed() - t << " s";
    WriteKaldiObject(Q,mat_wxfilename,binary);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
