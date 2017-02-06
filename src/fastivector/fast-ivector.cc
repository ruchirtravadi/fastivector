// fastivector/fast-ivector.cc

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

#include "fastivector/fast-ivector.h"

namespace kaldi {
  void FastIvectorDiagStats::AccStatsForUtterance(std::string utt, const Matrix<BaseFloat> &feats, const Posterior &post) {
    KALDI_ASSERT(post.size() == feats.NumRows());
    int32 num_frames = post.size();
    Vector<BaseFloat> N_utt(num_gauss_,kSetZero);
    Vector<BaseFloat> F_utt(num_gauss_*feat_dim_,kSetZero);
    for(int32 i = 0; i < num_frames; i++) {
      // Populate the statistics 
      SubVector<BaseFloat> frame(feats,i);
      for(int32 j = 0; j < post[i].size(); j++) {
        int32 c = post[i][j].first;
        double pc = post[i][j].second;
        Vector<BaseFloat> Ftc(frame);
        KALDI_ASSERT(c < num_gauss_);
        SubVector<BaseFloat> Fc(F_utt,c*feat_dim_,feat_dim_);
        Fc.AddVec(pc,Ftc);
        // Nc = Nc + pc
        N_utt(c) = N_utt(c) + pc;
      }
    }
    N_.AddVec(1.0,N_utt);
    F_.AddVec(1.0,F_utt);
    stats_N_writer_.Write(utt,N_utt);
    N_utt.ApplyFloor(1e-3);
    for(int32 c = 0; c < num_gauss_; c++) {
      SubVector<BaseFloat> Fc(F_utt,c*feat_dim_,feat_dim_);
      Fc.Scale(1.0/N_utt(c));
    }
    stats_F_writer_.Write(utt,F_utt);

    // Get the utterance specific covariance and counts
    for(int32 i = 0; i < num_frames; i++) {
      SubVector<BaseFloat> frame(feats,i);
      for(int32 j = 0; j < post[i].size(); j++) {
        int32 c = post[i][j].first;
        double pc = post[i][j].second;
        Vector<BaseFloat> Stc(frame);
        SubVector<BaseFloat> Fc(F_utt,c*feat_dim_,feat_dim_);
        Stc.AddVec(-1,Fc);
        S_[c].AddVec2(pc,Stc);
      }
    }
    // Increment the number of utterances
    num_utt_++;
  }

  void FastIvectorDiagStats::GetS(std::vector<Vector<BaseFloat> > &S) {
    S.resize(num_gauss_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S[c].Resize(S_[c].Dim());
      S[c].CopyFromVec(S_[c]);
      S[c].Scale(1.0/N_(c));
    }
  }

  void FastIvectorDiagStats::GetF(Vector<BaseFloat> &F) {
    F.Resize(num_gauss_*feat_dim_);
    F.CopyFromVec(F_);
    for(int32 c = 0; c < num_gauss_; c++) {
      SubVector<BaseFloat> Fc(F,c*feat_dim_,feat_dim_);
      if(N_(c) > 1e-3) Fc.Scale(1.0/N_(c));
      else             Fc.Scale(1000);
    }
  }

  void FastIvectorDiagStats::Write(const std::string &file, bool binary) const {
    Output out(file, binary, true);
    Write(out.Stream(),binary);
    out.Close();
  }

  void FastIvectorDiagStats::Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<FastIvectorDiagStats>");
    if (!binary) os << "\n";
    WriteToken(os,binary,"<U>");
    WriteBasicType(os,binary,num_utt_);
    WriteToken(os, binary, "<N>");
    N_.Write(os,binary);
    WriteToken(os, binary, "<F>");
    F_.Write(os,binary);
    WriteToken(os, binary, "<S>");
    for(int32 c = 0; c < num_gauss_; c++) {
      S_[c].Write(os,binary);
    }
    WriteToken(os, binary, "</FastIvectorDiagStats>");
  }

  void FastIvectorDiagStats::Read(const std::string &file) {
    bool binary;
    Input in(file, &binary);
    Read(in.Stream(), binary);
    in.Close();
  }

  void FastIvectorDiagStats::Read(std::istream &is, bool binary) {
    std::string token;
    ReadToken(is, binary, &token);
    if (token !="<FastIvectorDiagStats>") {
      KALDI_ERR << "Expected <FastIvectorDiagStats>, got " << token;
    }

    ReadToken(is, binary, &token);
    if (token !="<U>") {
      KALDI_ERR << "Expected <U>, got " << token;
    }
    ReadBasicType(is,binary,&num_utt_);

    ReadToken(is, binary, &token);
    if (token !="<N>") {
      KALDI_ERR << "Expected <N>, got " << token;
    }
    N_.Read(is,binary);
    num_gauss_ = N_.Dim();

    ReadToken(is, binary, &token);
    if (token !="<F>") {
      KALDI_ERR << "Expected <F>, got " << token;
    }
    F_.Read(is,binary);

    ReadToken(is, binary, &token);
    if (token !="<S>") {
      KALDI_ERR << "Expected <S>, got " << token;
    }
    S_.resize(num_gauss_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S_[c].Read(is,binary);
    }
    if(num_gauss_ > 0) feat_dim_ = S_[0].Dim();
    ReadToken(is, binary, &token);
    if (token !="</FastIvectorDiagStats>") {
      KALDI_ERR << "Expected </FastIvectorDiagStats>, got " << token;
    }
  }

  void FastIvectorDiagStats::Read(const std::string &file, Vector<BaseFloat> *N, Vector<BaseFloat> *F, std::vector<Vector<BaseFloat> > &S) {
    bool binary;
    Input in(file, &binary);
    std::istream &is(in.Stream());
    std::string token;
    ReadToken(is, binary, &token);
    if (token !="<FastIvectorDiagStats>") {
      KALDI_ERR << "Expected <FastIvectorDiagStats>, got " << token;
    }

    ReadToken(is, binary, &token);
    if (token !="<U>") {
      KALDI_ERR << "Expected <U>, got " << token;
    }
    ReadBasicType(is,binary,&num_utt_);

    ReadToken(is, binary, &token);
    if (token !="<N>") {
      KALDI_ERR << "Expected <N>, got " << token;
    }
    N->Read(is,binary);
    num_gauss_ = N->Dim();

    ReadToken(is, binary, &token);
    if (token !="<F>") {
      KALDI_ERR << "Expected <F>, got " << token;
    }
    F->Read(is,binary);

    ReadToken(is, binary, &token);
    if (token !="<S>") {
      KALDI_ERR << "Expected <S>, got " << token;
    }
    S.resize(num_gauss_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S[c].Read(is,binary);
    }
    if(num_gauss_ > 0) feat_dim_ = S[0].Dim();

    ReadToken(is, binary, &token);
    if (token !="</FastIvectorDiagStats>") {
      KALDI_ERR << "Expected </FastIvectorDiagStats>, got " << token;
    }   
    in.Close(); 
  }

  void FastIvectorDiagStats::AddStats(const FastIvectorDiagStats &other_stats) {
    num_utt_ += other_stats.num_utt_;
    N_.AddVec(1.0,other_stats.N_);
    F_.AddVec(1.0,other_stats.F_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S_[c].AddVec(1.0,other_stats.S_[c]);
    }
  }

  FastIvectorDiag::FastIvectorDiag(const Matrix<BaseFloat> &M, const std::vector<Vector<BaseFloat> >& S, 
                                   const Matrix<BaseFloat> &S_Inv_T, const Vector<BaseFloat> &D) : S_Inv_T_(S_Inv_T), M_(M), D_sq_(D) {
    int32 num_gauss = M.NumRows(), feat_dim = M.NumCols();
    KALDI_ASSERT(S.size() == num_gauss);
    S_.resize(num_gauss);
    for(int32 i = 0; i < S.size(); i++) {
      KALDI_ASSERT(S[i].Dim() == feat_dim);
      S_[i].Resize(feat_dim);
      S_[i].CopyFromVec(S[i]);
    }
    D_sq_.ApplyPow(2.0);
  }

  void FastIvectorDiag::Read(const std::string &file) {
    bool binary;
    Input in(file, &binary);
    Read(in.Stream(), binary);
    in.Close();
  }

  void FastIvectorDiag::Read(std::istream &is, bool binary) {
    std::string token;
    ReadToken(is, binary, &token);
    if (token !="<FastIvectorDiagModel>") {
      KALDI_ERR << "Expected <FastIvectorDiagModel>, got " << token;
    }
    ReadToken(is, binary, &token);
    if (token !="<M>") {
      KALDI_ERR << "Expected <M>, got " << token;
    }
    M_.Read(is,binary);
    ReadToken(is, binary, &token);
    if (token !="<S_inv_T>") {
      KALDI_ERR << "Expected <S_inv_T>, got " << token;
    }
    S_Inv_T_.Read(is,binary);
    ReadToken(is, binary, &token);
    if (token !="<S>") {
      KALDI_ERR << "Expected <S>, got " << token;
    }
    S_.resize(M_.NumRows());
    for(int32 c = 0; c < S_.size(); c++) {
      S_[c].Read(is,binary);
    }
    ReadToken(is, binary, &token);
    if (token !="<D>") {
      KALDI_ERR << "Expected <D>, got " << token;
    }
    D_sq_.Read(is,binary);
    ReadToken(is, binary, &token);
    if (token !="</FastIvectorDiagModel>") {
      KALDI_ERR << "Expected </FastIvectorDiagModel>, got " << token;
    }
  }

  void FastIvectorDiag::Write(const std::string &file, bool binary) const {
    Output out(file, binary, true);
    Write(out.Stream(), binary);
    out.Close();
  }

  void FastIvectorDiag::Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<FastIvectorDiagModel>");
    if (!binary) os << "\n";
    // Write the means
    WriteToken(os, binary, "<M>");
    M_.Write(os,binary);
    // Write the Subspace Matrix
    WriteToken(os, binary, "<S_inv_T>");
    S_Inv_T_.Write(os,binary);
    // Write the Covariance Matrices
    WriteToken(os,binary,"<S>");
    if (!binary) os << "\n";
    for(int32 c = 0; c < NumGauss() ; c++) {
      S_[c].Write(os,binary);
      if (!binary) os << "\n";
    }
    // Write the singular values
    WriteToken(os, binary, "<D>");
    D_sq_.Write(os,binary);
    WriteToken(os, binary, "</FastIvectorDiagModel>");
  }

  void FastIvectorDiag::GetIvector(const Matrix<BaseFloat> &feats, const Posterior &post, Vector<BaseFloat> &x)  {
    int32 ivec_dim = x.Dim();
    KALDI_ASSERT(ivec_dim <= IvecDim());
    // Accumulate the stats
    Vector<BaseFloat> F_utt(NumGauss()*FeatDim(),kSetZero), N_utt(NumGauss(),kSetZero);
    int32 num_frames = feats.NumRows(), feat_dim = FeatDim(), num_gauss = NumGauss();
    for(int32 i = 0; i < feats.NumRows(); i++) {
      // Populate the statistics 
      SubVector<BaseFloat> frame(feats,i);
      for(int32 j = 0; j < post[i].size(); j++) {
        int32 c = post[i][j].first;
        double pc = post[i][j].second;
        Vector<BaseFloat> Ftc(frame);
        Ftc.AddVec(-1,M_.Row(c));
        SubVector<BaseFloat> Fc(F_utt,c*feat_dim,feat_dim);
        Fc.AddVec(pc,Ftc);
        // Nc = Nc + pc
        N_utt(c) += pc;     
      }
    }
    // Normalize the F stats
    for(int32 c = 0; c < num_gauss; c++) {
      SubVector<BaseFloat> Fc(F_utt,c*feat_dim,feat_dim);
      if(N_utt(c) > 1e-6) {
        Fc.Scale(pow(N_utt(c),-0.5));
      } else {
        Fc.Scale(1000);
      }
    }
    x.SetZero();
    SubMatrix<BaseFloat> S_Inv_T_submat(S_Inv_T_,0,S_Inv_T_.NumRows(),0,ivec_dim);
    x.AddMatVec(1.0,S_Inv_T_submat,kTrans,F_utt,1.0);
    Vector<BaseFloat> x_inv_cov(ivec_dim);
    x_inv_cov.Set(1.0/num_frames);
    x_inv_cov.AddVec(1.0,D_sq_.Range(0,ivec_dim));
    x_inv_cov.InvertElements();
    x.MulElements(x_inv_cov);
    x.Scale(pow(num_frames,-0.5));
  }

  void FastIvectorDiag::GetSubSpace(Matrix<BaseFloat> *T, int32 c)  {
    int32 feat_dim = FeatDim(), ivec_dim = IvecDim();
    SubMatrix<BaseFloat> Tc(S_Inv_T_,c*feat_dim,feat_dim,0,ivec_dim);
    T->Resize(feat_dim,ivec_dim);
    T->SetZero();
    T->AddDiagVecMat(1.0,S_[c],Tc,kNoTrans,1.0);
  }

  void FastIvectorDiag::GetSubSpace(Matrix<BaseFloat> *T)  {
    T->Resize(S_Inv_T_.NumRows(),S_Inv_T_.NumCols(),kSetZero);
    int32 feat_dim = FeatDim(), ivec_dim = IvecDim(), num_gauss = NumGauss();
    for(int32 c = 0; c < num_gauss; c++) {
      SubMatrix<BaseFloat> S_Inv_Tc(S_Inv_T_,c*feat_dim,feat_dim,0,ivec_dim);
      SubMatrix<BaseFloat> Tc(*T,c*feat_dim,feat_dim,0,ivec_dim);
      Tc.AddDiagVecMat(1.0,S_[c],S_Inv_Tc,kNoTrans,1.0);
    }
  }

  void FastIvectorFullStats::AccStatsForUtterance(std::string utt, const Matrix<BaseFloat> &feats, const Posterior &post) {
    KALDI_ASSERT(post.size() == feats.NumRows());
    int32 num_frames = post.size();
    Vector<BaseFloat> N_utt(num_gauss_,kSetZero);
    Vector<BaseFloat> F_utt(num_gauss_*feat_dim_,kSetZero);
    for(int32 i = 0; i < num_frames; i++) {
      // Populate the statistics 
      SubVector<BaseFloat> frame(feats,i);
      for(int32 j = 0; j < post[i].size(); j++) {
        int32 c = post[i][j].first;
        double pc = post[i][j].second;
        // Ftc = (Xt - Mc)
        Vector<BaseFloat> Ftc(frame);
        KALDI_ASSERT(c < num_gauss_);
        SubVector<BaseFloat> Fc(F_utt,c*feat_dim_,feat_dim_);
        Fc.AddVec(pc,Ftc);
        // Nc = Nc + pc
        N_utt(c) = N_utt(c) + pc;
      }
    }
    N_.AddVec(1.0,N_utt);
    F_.AddVec(1.0,F_utt);
    stats_N_writer_.Write(utt,N_utt);
    N_utt.ApplyFloor(1e-3);
    for(int32 c = 0; c < num_gauss_; c++) {
      SubVector<BaseFloat> Fc(F_utt,c*feat_dim_,feat_dim_);
      Fc.Scale(1.0/N_utt(c));
    }
    stats_F_writer_.Write(utt,F_utt);

    // Get the utterance specific covariance and counts
    for(int32 i = 0; i < num_frames; i++) {
      SubVector<BaseFloat> frame(feats,i);
      for(int32 j = 0; j < post[i].size(); j++) {
        int32 c = post[i][j].first;
        double pc = post[i][j].second;
        Vector<BaseFloat> Stc(frame);
        SubVector<BaseFloat> Fc(F_utt,c*feat_dim_,feat_dim_);
        Stc.AddVec(-1,Fc);
        S_[c].AddVec2(pc,Stc);
      }
    }
    // Increment the number of utterances
    num_utt_++;
  }

  void FastIvectorFullStats::GetS(std::vector<SpMatrix<BaseFloat> > &S) {
    S.resize(num_gauss_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S[c].Resize(S_[c].NumRows());
      S[c].CopyFromSp(S_[c]);
      S[c].Scale(1.0/N_(c));
    }
  }

  void FastIvectorFullStats::GetF(Vector<BaseFloat> &F) {
    F.Resize(num_gauss_*feat_dim_);
    F.CopyFromVec(F_);
    for(int32 c = 0; c < num_gauss_; c++) {
      SubVector<BaseFloat> Fc(F,c*feat_dim_,feat_dim_);
      if(N_(c) > 1e-3) Fc.Scale(1.0/N_(c));
      else             Fc.Scale(1000);
    }
  }

  void FastIvectorFullStats::Write(const std::string &file, bool binary) const {
    Output out(file, binary, true);
    Write(out.Stream(),binary);
    out.Close();
  }

  void FastIvectorFullStats::Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<FastIvectorFullStats>");
    if (!binary) os << "\n";
     WriteToken(os,binary,"<U>");
    WriteBasicType(os,binary,num_utt_);
    WriteToken(os, binary, "<N>");
    N_.Write(os,binary);
    WriteToken(os, binary, "<F>");
    F_.Write(os,binary);
    WriteToken(os, binary, "<S>");
    for(int32 c = 0; c < num_gauss_; c++) {
      S_[c].Write(os,binary);
    }
    WriteToken(os, binary, "</FastIvectorFullStats>");
  }

  void FastIvectorFullStats::Read(const std::string &file) {
    bool binary;
    Input in(file, &binary);
    Read(in.Stream(), binary);
    in.Close();
  }

  void FastIvectorFullStats::Read(std::istream &is, bool binary) {
    std::string token;
    ReadToken(is, binary, &token);
    if (token !="<FastIvectorFullStats>") {
      KALDI_ERR << "Expected <FastIvectorFullStats>, got " << token;
    }

    ReadToken(is, binary, &token);
    if (token !="<U>") {
      KALDI_ERR << "Expected <U>, got " << token;
    }
    ReadBasicType(is,binary,&num_utt_);

    ReadToken(is, binary, &token);
    if (token !="<N>") {
      KALDI_ERR << "Expected <N>, got " << token;
    }
    N_.Read(is,binary);
    num_gauss_ = N_.Dim();

    ReadToken(is, binary, &token);
    if (token !="<F>") {
      KALDI_ERR << "Expected <F>, got " << token;
    }
    F_.Read(is,binary);

    ReadToken(is, binary, &token);
    if (token !="<S>") {
      KALDI_ERR << "Expected <S>, got " << token;
    }
    S_.resize(num_gauss_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S_[c].Read(is,binary);
    }
    if(num_gauss_ > 0) feat_dim_ = S_[0].NumRows();

    ReadToken(is, binary, &token);
    if (token !="</FastIvectorFullStats>") {
      KALDI_ERR << "Expected </FastIvectorFullStats>, got " << token;
    }
  }

  void FastIvectorFullStats::Read(const std::string &file, Vector<BaseFloat> *N, Vector<BaseFloat> *F, std::vector<SpMatrix<BaseFloat> > &S) {
    bool binary;
    Input in(file, &binary);
    std::istream &is(in.Stream());
    std::string token;
    ReadToken(is, binary, &token);
    if (token !="<FastIvectorFullStats>") {
      KALDI_ERR << "Expected <FastIvectorFullStats>, got " << token;
    }

    ReadToken(is, binary, &token);
    if (token !="<U>") {
      KALDI_ERR << "Expected <U>, got " << token;
    }
    ReadBasicType(is,binary,&num_utt_);

    ReadToken(is, binary, &token);
    if (token !="<N>") {
      KALDI_ERR << "Expected <N>, got " << token;
    }
    N->Read(is,binary);
    num_gauss_ = N->Dim();

    ReadToken(is, binary, &token);
    if (token !="<F>") {
      KALDI_ERR << "Expected <F>, got " << token;
    }
    F->Read(is,binary);

    ReadToken(is, binary, &token);
    if (token !="<S>") {
      KALDI_ERR << "Expected <S>, got " << token;
    }
    S.resize(num_gauss_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S[c].Read(is,binary);
    }
    if(num_gauss_ > 0) feat_dim_ = S[0].NumRows();

    ReadToken(is, binary, &token);
    if (token !="</FastIvectorFullStats>") {
      KALDI_ERR << "Expected </FastIvectorFullStats>, got " << token;
    }
    in.Close();
  }

  void FastIvectorFullStats::AddStats(const FastIvectorFullStats &other_stats) {
    num_utt_ += other_stats.num_utt_;
    N_.AddVec(1.0,other_stats.N_);
    F_.AddVec(1.0,other_stats.F_);
    for(int32 c = 0; c < num_gauss_; c++) {
      S_[c].AddSp(1.0,other_stats.S_[c]);
    }
  }

  FastIvectorFull::FastIvectorFull(const Matrix<BaseFloat> &M, const std::vector<SpMatrix<BaseFloat> >& S,
                                   const Matrix<BaseFloat> &S_Inv_T, const Vector<BaseFloat> &D) : S_Inv_T_(S_Inv_T), M_(M), D_sq_(D)  {
    int32 num_gauss = M_.NumRows(), feat_dim = M_.NumCols();
    KALDI_ASSERT(S.size() == num_gauss);
    S_.resize(num_gauss);
    for(int32 i = 0; i < S.size(); i++) {
      S_[i].Resize(feat_dim);
      S_[i].CopyFromSp(S[i]);
    }
    D_sq_.ApplyPow(2.0);
  }

  void FastIvectorFull::Read(const std::string &file) {
    bool binary;
    Input in(file, &binary);
    Read(in.Stream(), binary);
    in.Close();
  }

  void FastIvectorFull::Read(std::istream &is, bool binary) {
    std::string token;
    ReadToken(is, binary, &token);
    if (token !="<FastIvectorFullModel>") {
      KALDI_ERR << "Expected <FastIvectorFullModel>, got " << token;
    }
    ReadToken(is, binary, &token);
    if (token !="<M>") {
      KALDI_ERR << "Expected <M>, got " << token;
    }
    M_.Read(is,binary);
    ReadToken(is, binary, &token);
    if (token !="<S_inv_T>") {
      KALDI_ERR << "Expected <S_inv_T>, got " << token;
    }
    S_Inv_T_.Read(is,binary);
    ReadToken(is, binary, &token);
    if (token !="<S>") {
      KALDI_ERR << "Expected <S>, got " << token;
    }
    S_.resize(NumGauss());
    for(int32 c = 0; c < S_.size(); c++) {
      S_[c].Read(is,binary);
    }
    ReadToken(is, binary, &token);
    if (token !="<D>") {
      KALDI_ERR << "Expected <D>, got " << token;
    }
    D_sq_.Read(is,binary);
    ReadToken(is, binary, &token);
    if (token !="</FastIvectorFullModel>") {
      KALDI_ERR << "Expected </FastIvectorFullModel>, got " << token;
    }
  }

  void FastIvectorFull::Write(const std::string &file, bool binary) const {
    Output out(file, binary, true);
    Write(out.Stream(), binary);
    out.Close();
  }

  void FastIvectorFull::Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<FastIvectorFullModel>");
    if (!binary) os << "\n";
    // Write the Means
    WriteToken(os,binary,"<M>"); 
    M_.Write(os,binary);
    // Write the Subspace Matrix
    WriteToken(os, binary, "<S_inv_T>");
    S_Inv_T_.Write(os,binary);
    // Write the Covariance Matrices
    WriteToken(os,binary,"<S>");
    if (!binary) os << "\n";
    for(int32 c = 0; c < NumGauss() ; c++) {
      S_[c].Write(os,binary);
      if (!binary) os << "\n";
    }
    // Write the singular values
    WriteToken(os, binary, "<D>");
    D_sq_.Write(os,binary);
    WriteToken(os, binary, "</FastIvectorFullModel>");
  }

  void FastIvectorFull::GetIvector(const Matrix<BaseFloat> &feats, const Posterior &post, Vector<BaseFloat> &x)  {
    int32 ivec_dim = x.Dim();
    KALDI_ASSERT(ivec_dim <= IvecDim());
    // Accumulate the stats
    Vector<BaseFloat> F_utt(NumGauss()*FeatDim()), N_utt(NumGauss(),kSetZero);
    int32 num_frames = feats.NumRows(), feat_dim = FeatDim(), num_gauss = NumGauss();
    for(int32 i = 0; i < feats.NumRows(); i++) {
      // Populate the statistics 
      SubVector<BaseFloat> frame(feats,i);
      for(int32 j = 0; j < post[i].size(); j++) {
        int32 c = post[i][j].first;
        double pc = post[i][j].second;
        // Ftc = (Xt - Mc)
        Vector<BaseFloat> Ftc(frame);
        Ftc.AddVec(-1,M_.Row(c));
        // Fc = Fc + pc*Ftc
        SubVector<BaseFloat> Fc(F_utt,c*feat_dim,feat_dim);
        Fc.AddVec(pc,Ftc);
        // Nc = Nc + pc
        N_utt(c) += pc;
      }
    }
    // Normalize the F stats
    for(int32 c = 0; c < num_gauss; c++) {
      SubVector<BaseFloat> Fc(F_utt,c*feat_dim,feat_dim);
      if(N_utt(c) > 1e-6) {
        Fc.Scale(pow(N_utt(c),-0.5));
      } else {
        Fc.Scale(1000);
      }
    }
    x.SetZero();
    SubMatrix<BaseFloat> S_Inv_T_submat(S_Inv_T_,0,S_Inv_T_.NumRows(),0,ivec_dim);
    x.AddMatVec(1.0,S_Inv_T_submat,kTrans,F_utt,1.0);
    Vector<BaseFloat> x_inv_cov(ivec_dim);
    x_inv_cov.Set(1.0/num_frames);
    x_inv_cov.AddVec(1.0,D_sq_.Range(0,ivec_dim));
    x_inv_cov.InvertElements();
    x.MulElements(x_inv_cov);
    x.Scale(pow(num_frames,-0.5));
  }

  void FastIvectorFull::GetSubSpace(Matrix<BaseFloat> *T, int32 c)  {
    int32 feat_dim = FeatDim(), ivec_dim = IvecDim();
    SubMatrix<BaseFloat> Tc(S_Inv_T_,c*feat_dim,feat_dim,0,ivec_dim);
    T->Resize(feat_dim,ivec_dim);
    T->SetZero();
    T->AddSpMat(1.0,S_[c],Tc,kNoTrans,1.0);
  }

  void FastIvectorFull::GetSubSpace(Matrix<BaseFloat> *T)  {
    T->Resize(S_Inv_T_.NumRows(),S_Inv_T_.NumCols(),kSetZero);
    int32 feat_dim = FeatDim(), ivec_dim = IvecDim(), num_gauss = NumGauss();
    for(int32 c = 0; c < num_gauss; c++) {
      SubMatrix<BaseFloat> S_Inv_Tc(S_Inv_T_,c*feat_dim,feat_dim,0,ivec_dim);
      SubMatrix<BaseFloat> Tc(*T,c*feat_dim,feat_dim,0,ivec_dim);
      Tc.AddSpMat(1.0,S_[c],S_Inv_Tc,kNoTrans,1.0);
    }
  }
}
