# LPBoost-Using-String-and-Fisher-Features

Yuxin Sun, University College London, London, UK, 2016.

Linear programming boosting (LPBoost) with string and Fisher features.

The repository has been tested to work under Python 2.7.

This repository provides code and data sets that have been used.

**Dependencies**

This repository has been successfully tested to work under Python 2.7.

The required dependencies to work with the repository are cvxopt >= 1.1.8, NumPy >= 1.6.1 and Scipy >= 0.9.

**Data**

Data/OVA_CFA_P277/: 11 subsets of immunised data that were used in the experiments. Each subset consists of 50000 randomly selected CDR3 sequences.

Data/Expanded: expanded CDR3 sequences in OVA immunised and CFA immunised mice at early stage (sampled at day 5, 7 or 14).

###Modules

Features/process_data.py: process CDR3 sequences, read and write .cPickle files for further use.

Features/generate_features.py: generate either string features or Fisher features.

######LPBoost/lpboost

a Python module for linear programming boosting (LPBoost).

**Test**

To test the code, one could simply run sample_code.py. This file would generate features from CDR3 sequences and perform LPBoost using the feature matrix that has been generated.
