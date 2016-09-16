// fastivectorbin/fast-ivector-diag-combine-stats.cc

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
      "Obta for a diagonal covariance fast ivector model\n"
      "Usage:  fast-ivector-diag-combine-stats [options] <stats.1> <stats.2> ... <stats.n> <stats-wxfilename> \n"
      "e.g.: fast-ivector-diag-combine-stats stats.1 stats.2 NS.stats\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string stats_outfilename = po.GetArg(po.NumArgs());
    int32 num_stats_files = po.NumArgs() - 1;
    // Read the first stats file
    FastIvectorDiagStats stats_out;
    stats_out.Read(po.GetArg(1));
    // Combine with rest of the stats files
    for(size_t i = 2; i <= num_stats_files; i++) {
      std::string stats_rxfilename = po.GetArg(i);
      FastIvectorDiagStats stats_in;
      stats_in.Read(stats_rxfilename);
      stats_out.AddStats(stats_in);
    }
    // Write
    stats_out.Write(stats_outfilename,binary);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
