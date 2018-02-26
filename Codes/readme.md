# Notes

Billing code prediction on mimic

# Differences between CUIs vs tokens as input

There are two functions for reading tokens and cuis respectively:

* read_ngrams()
* read_cuis()

Other functions depend on them:

* make_and_write_token_alphabet() uses read_cuis()
* load() uses read_cuis

# Todo

Make a super class for dataset and have two derived classes.
One for working with cuis and another for working with tokens.

# Results

75 epochs with embeddings

macro average p = 0.515948620205
macro average r = 0.445234742802
macro average f1 = 0.472892203829

75 epochs no embeddings

macro average p = 0.524717495517
macro average r = 0.404569707237
macro average f1 = 0.446981740939

