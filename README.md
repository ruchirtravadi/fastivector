# fastivector
This repository contains the code for fast ivector model training and ivector extraction in Kaldi using randomized SVD. 

## Theoretical details
The theory behind this technique is described in [this paper](Paper.pdf).
The randomized SVD algorithm we used is based on the proto algorithm in [this paper](http://users.cms.caltech.edu/~jtropp/papers/HMT11-Finding-Structure-SIREV.pdf).

## Using the code and scripts
### Compilation
1. This code is based on [Kaldi](http://kaldi-asr.org/), so you will need to install and compile it first.

2. Copy the  contents of [src](src) folder to the *src* folder of your Kaldi installation.

3. Compile the code:  
    ```bash
    # Navigate to <kaldi-root>/src
    cd ~/kaldi-trunk/src
    
    # Compile fastivector
    cd fastivector
    make depend
    make
    
    # Compile fastivectorbin
    cd ../fastivectorbin
    make depend
    make
    ```

### Scripts
Scripts are provided for both a [diagonal covariance](steps/fastivec_diag) and a [full covariance](steps/fastivec_full) version of the model. Just copy the contents of the required version inside the *steps* directory of the relevant *egs* directory (for example : \<kaldi-root\>/egs/wsj/s5/steps).

Then, the code can be executed by calling the top-level training/extraction script.

```bash
cd ~/kaldi-trunk/egs/wsj/s5
# Training
bash steps/fastivec_diag/train_fastivec_diag.sh --nj 4 data/train_si284 512 exp/fastivec_diag_512
# Extraction
bash steps/fastivec_diag/extract_ivec.sh --nj 4 data/test_eval92 exp/fastivec_diag_512  exp/fastivec_diag_512/ivectors_test_eval92
```
