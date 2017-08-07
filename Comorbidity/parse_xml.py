#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
import xml.etree.ElementTree as et
import os.path

# BASE = '/Users/Dima/Loyola/Data/Comorbidity/'
# NOTES = 'Xml/obesity_patient_records_training.xml'
# ANNOT = 'Xml/obesity_standoff_annotations_training.xml'
# OUTDIR = 'Text/Train/'

def parse_standoff(xml, disease, task):
  """Make patient to class mapping"""

  doc2label = {} # key: doc id, value: label
  tree = et.parse(xml)

  for task_elem in tree.iter('diseases'):
    if task_elem.attrib['source'] == task:
      for disease_elem in task_elem:
        if disease_elem.attrib['name'] == disease:
          for doc_elem in disease_elem:
            id = doc_elem.attrib['id']
            label = doc_elem.attrib['judgment']
            doc2label[id] = label

  return doc2label

def write_notes_to_files(xml, outdir):
  """Extract notes from xml and write to files"""

  tree = et.parse(xml)

  for doc in tree.iter('doc'):
    doc_id = doc.attrib['id']
    doc_text = doc[0].text

    file_name = os.path.join(base, outdir, '%s.txt' % doc_id)
    out_file = open(file_name, 'w')
    out_file.write(doc_text)

if __name__ == "__main__":

  # parse_standoff('Asthma', 'textual')
  write_notes_to_files()
