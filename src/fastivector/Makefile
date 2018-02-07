
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

OBJFILES = fast-ivector.o

LIBNAME = kaldi-fast-ivector

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
    ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
    ../util/kaldi-util.a ../ivector/kaldi-ivector.a

include ../makefiles/default_rules.mk
