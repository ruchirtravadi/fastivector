
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = fast-ivector-diag-acc-stats fast-ivector-diag-combine-stats fast-ivector-diag-randsvd-get-y fast-ivector-diag-randsvd-power-iter fast-ivector-diag-randsvd-get-q fast-ivector-diag-randsvd-get-b fast-ivector-diag-randsvd-est fast-ivector-diag-extract fast-ivector-full-acc-stats fast-ivector-full-combine-stats fast-ivector-full-randsvd-get-y fast-ivector-full-randsvd-power-iter fast-ivector-full-randsvd-get-q fast-ivector-full-randsvd-get-b fast-ivector-full-randsvd-est fast-ivector-full-extract

OBJFILES =

TESTFILES =


ADDLIBS = ../ivector/kaldi-ivector.a ../fastivector/kaldi-fast-ivector.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a \
    ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
    ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
