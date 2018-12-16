# Style Classifier for Plug-and-Play model

## Prerequisites

1. Python 2.7 or higher
2. tensorflow-gpu 1.0.0 or 1.1.0

## Cclczone this repository
`git clone https://github.com/adfsghjalison/PPGN_Style_Classifier.git`

### Data
`mkdir data`  
`mkdir data/data_[database_name]`  
1. Put training data `source_train` and testing data `source_test` in `data/data_[database_name]`  
format : one data a line  
[style label] +++$+++ [sentence]  

2. Put the word dictionary `dict` in `data/data_[database_name]`  
`dict` : a json file with  
`word : word_id`  
with `__BOS__`, `__EOS__`, `__UNK__`  

### Train
`python main.py --mode train`

### Test
`python main.py --mode test`

### Important Hyperparameters of the flags.py
`data_name` : database name  
`batch_size` : batch size  
`embedding_dim` : embedding layer dimension  
`unit_size` : latent dimension of seq2seq  
`max_length` : max length of input and output sentence  

## Files

### Folders
`data/` : training data / testing data / dictionary file  
`model/` : saved trained models  

### Files
`flags.py` : all settings  
`model.py` : model architecture  
`main.py` : main function  

