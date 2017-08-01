#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
import xml.etree.ElementTree as et
import os.path

BASE = '/Users/Dima/Loyola/Data/Comorbidity'
NOTES = 'Xml/obesity_patient_records_training.xml'
OUTDIR = 'Text/'

def write_notes_to_files():
  """Extract notes from xml and write to files"""

  xml_file = os.path.join(BASE, NOTES)
  tree = et.parse(xml_file)

  for doc in tree.iter('doc'):
    doc_id = doc.attrib['id']
    doc_text = doc[0].text

    file_name = os.path.join(BASE, OUTDIR, '%s.txt' % doc_id)
    out_file = open(file_name, 'w')
    out_file.write(doc_text)

if __name__ == "__main__":

  write_notes_to_files()
