=== sub-challenge 1 ===

train_cli.tsv: this file contains clinical information (gender and msi status) for the 80 training samples. 
train_pro.tsv: proteomics data based on spectral counting. Each row represents a protein and each column represents
               a training sample. 
sum_tab_1.csv: mislabeling information for the training samples. 1: clinical and 
               proteomics data not from the same sample (mismatch), 0: both data from the same sample (match).
test_cli.tsv:  this file contains clinical information (gender and msi status) for the 80 testing samples. 
test_pro.tsv:  similar to train_pro.txt, but for testing samples.
