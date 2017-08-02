#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
import xml.etree.ElementTree as et
import os.path

BASE = '/Users/Dima/Loyola/Data/Comorbidity/'
NOTES = 'Xml/obesity_patient_records_training.xml'
ANNOT = 'Xml/obesity_standoff_annotations_training.xml'
OUTDIR = 'Text/Train/'

def parse_standoff(disease, task):
  """Make patient to class mapping"""

  doc2label = {} # key: doc id, value: label

  xml_file = os.path.join(BASE, ANNOT)
  tree = et.parse(xml_file)

  for task_elem in tree.iter('diseases'):
    if task_elem.attrib['source'] == task:
      for disease_elem in task_elem:
        if disease_elem.attrib['name'] == disease:
          for doc_elem in disease_elem:
            id = doc_elem.attrib['id']
            label = doc_elem.attrib['judgment']
            doc2label[id] = label

  return doc2label

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

  # parse_standoff('Asthma', 'textual')
  write_notes_to_files()
