// fastivector/fast-ivector.h

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

#ifndef KALDI_FAST_IVECTOR_H_
#define KALDI_FAST_IVECTOR_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"

namespace kaldi {

  // Model estimation options
  struct FastIvectorEstimationOptions {
    int32 ivec_dim, p, batch_size;
    FastIvectorEstimationOptions(): ivec_dim(400), p(100), batch_size(100) {}
    void Register(OptionsItf *po) {
      po->Register("ivec-dim", &ivec_dim,"ivector subspace dimension");
      po->Register("p",&p,"Oversampling parameter p used for randomized SVD");
      po->Register("batch-size",&batch_size, "batch size for processing statistics (used for memory consumption control)");
    }
  };

  // Class to hold accumulated statistics for diagonal covariance ivector model estimation
  class FastIvectorDiagStats {
    public:
      // Constructors
      FastIvectorDiagStats() {}
      FastIvectorDiagStats(int32 num_gauss,int32 feat_dim,
                           std::string stats_N_wspecifier, 
                           std::string stats_F_wspecifier) : 
        num_gauss_(num_gauss), feat_dim_(feat_dim), num_utt_(0),
        stats_F_wspecifier_(stats_F_wspecifier),
        stats_N_wspecifier_(stats_N_wspecifier) {
        N_.Resize(num_gauss_);
        F_.Resize(num_gauss_*feat_dim_,kSetZero);
        S_.resize(num_gauss_,Vector<BaseFloat>(feat_dim_,kSetZero));
        stats_N_writer_.Open(stats_N_wspecifier_),
        stats_F_writer_.Open(stats_F_wspecifier_);
      }
      
      // Read the accumulated stats
      void Read(const std::string &file);
      void Read(std::istream &is, bool binary);
      // Read directly into specified pointers
      void Read(const std::string &file, Vector<BaseFloat> *N, Vector<BaseFloat> *F, std::vector<Vector<BaseFloat> > &S);
      // Write the accumulated stats (utterance stats are written separately)
      void Write(const std::string &file, bool binary) const;
      void Write(std::ostream &os, bool binary) const;

      // Access the stats
      int64 GetNumUtt() { return num_utt_; }
      void GetN(Vector<BaseFloat> *N) { N->Resize(N_.Dim()); N->CopyFromVec(N_); }
      BaseFloat GetN(int32 c) { return N_(c);}
      void GetF(Vector<BaseFloat> &F);
      void GetS(std::vector<Vector<BaseFloat> > &S);
      void GetS(Vector<BaseFloat> *Sc, int32 c) { Sc->Resize(S_[c].Dim()); Sc->CopyFromVec(S_[c]); Sc->Scale(1.0/N_(c));}

      // Access Counts/Dimensions
      int32 NumGauss() { return num_gauss_; }
      int32 FeatDim() { return feat_dim_; }

      // Accumulate stats for an utterance
      void AccStatsForUtterance(std::string utt, const Matrix<BaseFloat> &feats, const Posterior &post);

      // Combine stats from another object
      void AddStats(const FastIvectorDiagStats &other_stats);

    private:
      // Dimension values to be used in the model
      int32 num_gauss_,feat_dim_;
      // Number of utterances for which stats are accumulated
      int64 num_utt_;
      // Sample covariance matrices for each component
      std::vector<Vector<BaseFloat> > S_;
      // Zeroth, first order stats
      Vector<BaseFloat> N_, F_;
      // Writers for zeroth, first order stats
      std::string stats_F_wspecifier_, stats_N_wspecifier_;
      BaseFloatVectorWriter stats_F_writer_, stats_N_writer_;
  };

  // Diagonal Covariance Fast Ivector Model Class
  class FastIvectorDiag{
    public:
      // Constructors
      // Default
      FastIvectorDiag() {};
      // From SVD
      FastIvectorDiag(const Matrix<BaseFloat> &M, const std::vector<Vector<BaseFloat> >& S_,
                                   const Matrix<BaseFloat> &S_Inv_T, const Vector<BaseFloat> &D);

      // Return the estimated i-vector from provided features and posteriors
      void GetIvector(const Matrix<BaseFloat> &feats, const Posterior &post, Vector<BaseFloat> &x);

      // Return the entire subspace matrix
      void GetSubSpace(Matrix<BaseFloat> *T);
      // Return a component-specific subspace matrix
      void GetSubSpace(Matrix<BaseFloat> *T, int32 c);
      // Return all covariance matrices
      void GetCovariance(std::vector<Vector<BaseFloat> > *S);
      // Return a component specific covariance matrix
      void GetCovariance(Vector<BaseFloat> *S, int32 c) { S->Resize(FeatDim()); S->CopyFromVec(S_[c]); }
      // Returns the number of mixture components in the UBM
      int32 NumGauss() const { return M_.NumRows(); }
      // Returns the dimensionality of the mean vectors
      int32 FeatDim() const { return M_.NumCols(); }
      // Returns the ivector dimension
      int32 IvecDim() const {return S_Inv_T_.NumCols(); }

      // IO functions
      void Read(const std::string &file);
      void Read(std::istream &is, bool binary);
      void Write(const std::string &file, bool binary) const;
      void Write(std::ostream &os, bool binary) const;

    private:
      // (Transformed) Subspace Matrix
      Matrix<BaseFloat> S_Inv_T_;
      // Componentwise Covariance Matrices (Diagonal)
      std::vector<Vector<BaseFloat> > S_;
      // UBM Means
      Matrix<BaseFloat> M_;
      // Squared singular values, used for efficient approximate ivector estimation
      Vector<BaseFloat> D_sq_;
  };

  // Class to hold accumulated statistics for full covariance ivector model estimation
  class FastIvectorFullStats {
    public:
      // Constructors
      FastIvectorFullStats() {}
      FastIvectorFullStats(int32 num_gauss, int32 feat_dim, 
                           std::string stats_N_wspecifier,
                           std::string stats_F_wspecifier) :
        num_gauss_(num_gauss), feat_dim_(feat_dim), num_utt_(0), 
        stats_N_wspecifier_(stats_N_wspecifier),
        stats_F_wspecifier_(stats_F_wspecifier) {
        N_.Resize(num_gauss_);
        F_.Resize(num_gauss_*feat_dim_,kSetZero);
        S_.resize(num_gauss_,SpMatrix<BaseFloat>(feat_dim_,kSetZero));
        stats_N_writer_.Open(stats_N_wspecifier_);
        stats_F_writer_.Open(stats_F_wspecifier_);
      }

      // IO functions
      // Read the accumulated stats
      void Read(const std::string &file);
      void Read(std::istream &is, bool binary);
      // Read directly into specified pointers
      void Read(const std::string &file, Vector<BaseFloat> *N, Vector<BaseFloat> *F, std::vector<SpMatrix<BaseFloat> > &S);
      // Write the accumulated stats (utterance stats are written separately)
      void Write(const std::string &file, bool binary) const;
      void Write(std::ostream &os, bool binary) const;

      // Access the stats
      int64 GetNumUtt() { return num_utt_; }
      void GetN(Vector<BaseFloat> *N) { N->Resize(N_.Dim()); N->CopyFromVec(N_); }
      BaseFloat GetN(int32 c) { return N_(c);}
      void GetF(Vector<BaseFloat> &F);
      void GetS(std::vector<SpMatrix<BaseFloat> > &S);
      void GetS(SpMatrix<BaseFloat> *Sc, int32 c) { Sc->Resize(S_[c].NumRows()); Sc->CopyFromSp(S_[c]); Sc->Scale(1.0/N_(c));}

      // Access Counts/Dimensions
      int32 NumGauss() { return num_gauss_; }
      int32 FeatDim() { return feat_dim_; }

      // Accumulate stats for an utterance
      void AccStatsForUtterance(std::string utt, const Matrix<BaseFloat> &feats, const Posterior &post);

      // Combine stats from another object
      void AddStats(const FastIvectorFullStats &other_stats);

    private:
      // Dimension values to be used in the model
      int32 num_gauss_,feat_dim_;
      // Number of utterances for which stats are accumulated
      int64 num_utt_;
      // Sample covariance matrices for each component
      std::vector<SpMatrix<BaseFloat> > S_;
      // Zeroth, first order stats
      Vector<BaseFloat> N_, F_;
      // Writer for zeroth, first order stats
      std::string stats_N_wspecifier_,stats_F_wspecifier_;
      BaseFloatVectorWriter stats_N_writer_,stats_F_writer_;
  };


  // Full Covariance Fast Ivector Model Class
  class FastIvectorFull{
    public:
      // Constructors
      // Default
      FastIvectorFull() {};
      // From SVD
      FastIvectorFull(const Matrix<BaseFloat> &M, const std::vector<SpMatrix<BaseFloat> >& S_,
                                   const Matrix<BaseFloat> &S_Inv_T, const Vector<BaseFloat> &D);

      // Return the estimated i-vector from provided features and posteriors
      void GetIvector(const Matrix<BaseFloat> &feats, const Posterior &post, Vector<BaseFloat> &x);

      // Return the entire subspace matrix
      void GetSubSpace(Matrix<BaseFloat> *T);
      // Return a component-specific subspace matrix
      void GetSubSpace(Matrix<BaseFloat> *T, int32 c);
      // Return all covariance matrices
      void GetCovariance(std::vector<SpMatrix<BaseFloat> > *S);
      // Return a component specific covariance matrix
      void GetCovariance(SpMatrix<BaseFloat> *S, int32 c) { S->Resize(FeatDim()); S->CopyFromSp(S_[c]); }
      // Returns the number of mixture components in the UBM
      int32 NumGauss() const { return M_.NumRows(); }
      // Returns the dimensionality of the mean vectors
      int32 FeatDim() const { return M_.NumCols(); }
      // Returns the ivector dimension
      int32 IvecDim() const {return S_Inv_T_.NumCols(); }

      // IO functions
      void Read(const std::string &file);
      void Read(std::istream &is, bool binary);
      void Write(const std::string &file, bool binary) const;
      void Write(std::ostream &os, bool binary) const;

    private:
      // (Transformed) Subspace Matrix
      Matrix<BaseFloat> S_Inv_T_;
      // Componentwise Covariance Matrices
      std::vector<SpMatrix<BaseFloat> > S_;
      // UBM Means
      Matrix<BaseFloat> M_;
      // Squared singular values, used for efficient approximate ivector estimation
      Vector<BaseFloat> D_sq_;
  };
}  // namespace kaldi

#endif // KALDI_FAST_IVECTOR_H_
