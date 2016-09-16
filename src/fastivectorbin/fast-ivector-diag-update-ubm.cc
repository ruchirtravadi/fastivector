// fastivectorbin/fast-ivector-diag-update-ubm.cc

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
      "Update a UBM model by replacing variances from a Fast Ivector Model\n"
      "Usage:  fast-ivector-diag-update-ubm [options] <diag-gmm-in> <fastivec-mdl> <diag-gmm-out>\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string diag_gmm_rxfilename = po.GetArg(1),
                fastivec_mdl_rxfilename = po.GetArg(2),
                diag_gmm_wxfilename = po.GetArg(3);
    
    // Create and write the ivector model
    DiagGmm diag_gmm;
    ReadKaldiObject(diag_gmm_rxfilename, &diag_gmm);
    FastIvectorDiag FastIvec_Model;
    FastIvec_Model.Read(fastivec_mdl_rxfilename);
    Matrix<BaseFloat> Inv_Vars(diag_gmm.NumGauss(),diag_gmm.Dim());
    for(int32 c = 0; c < diag_gmm.NumGauss(); c++) {
      Vector<BaseFloat> Inv_Sc;
      FastIvec_Model.GetCovariance(&Inv_Sc,c);
      Inv_Vars.Row(c).CopyFromVec(Inv_Sc);
      Inv_Vars.Row(c).InvertElements();
    }
    diag_gmm.SetInvVars(Inv_Vars);
    diag_gmm.ComputeGconsts();
    WriteKaldiObject(diag_gmm,diag_gmm_wxfilename,binary);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
