# General

Code for applying learned representations to comorbodity data

# Tokens vs CUIs as input

* In dataset.py, loop over file_feat_list instead of set(file_feat_list)
* In dense.cfg change maxlen to 9995 instead of 1387 (token seqs are longer)

