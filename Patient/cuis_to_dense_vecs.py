#!/usr/bin/env python

from keras.models import load_model
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import configparser, os, pickle
import sys

''' Purpose: Taking a file that has CUIs per patient and loading a neural model
    that maps from CUIs to dense vectors, and writing out a vector per patient.
    Argument: <config file> with a [data] section and model_file and
    alphabet_pickle sections. Reads a patient-cui file from standard input. The
    format for that file should be one patient per line:
    <pt id>: <cui_0> <cui_1> ... <cui_N>
    where cuis are space-separated and unique.
'''

def main(args):
    if len(args) < 1:
        sys.stderr.write("One required argument: <config file>\n")
        sys.exit(-1)

    cfg = configparser.ConfigParser()
    cfg.read(args[0])
    maxlen = 1535

    # load pre-trained model
    model = load_model(cfg.get('data', 'model_file'))
    interm_layer_model = Model(
        inputs=model.input,
        outputs=model.get_layer('HL').output)
    
    alphabet_pickle = cfg.get('data', 'alphabet_pickle')
    pkl = open(alphabet_pickle, 'rb')
    token2int = pickle.load(pkl)

    for line in sys.stdin:
        line = line.rstrip()
        ptid, cui_str = line.split(":")
        sys.stderr.write("Reading cuis for patient %s\n" % (ptid))
        cuis = cui_str.split(" ")
        example = []
        for token in cuis:
            if token in token2int:
                example.append(token2int[token])
            else:
                example.append(token2int['oov_word'])
        
        x_train = pad_sequences([example], maxlen=maxlen)
        x_vecs = interm_layer_model.predict(x_train)
        vec_str = ' '.join([str(f) for f in x_vecs[0].tolist()])
        print("%s: %s" % (ptid, vec_str))
        #print("Finished training patient %s" % (ptid))

if __name__ == '__main__':
    main(sys.argv[1:])
