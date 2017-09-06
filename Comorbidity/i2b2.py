#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
import xml.etree.ElementTree as et
import os.path

# map labels so this is a binary task
to_binary = {'Y': 'Yes', 'N': 'No', 'Q': 'No', 'U': 'No'}
to_int = {'Y': 1, 'N': 0, 'Q': 0, 'U': 0}

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
            binary_label = to_binary[label]
            doc2label[id] = binary_label

  return doc2label

def parse_standoff_vectorized(xml, task, exclude=set()):
  """Map each patient to a vector of labels"""

  # map diseases to integers
  diseases = get_disease_names(xml, exclude)
  dis2int = dict([[d, i] for i, d in enumerate(diseases)])

  doc2labels = {} # key: doc id, value: vector of labels
  tree = et.parse(xml)

  for task_elem in tree.iter('diseases'):
    if task_elem.attrib['source'] == task:
      for disease_elem in task_elem:

        disease_name = disease_elem.attrib['name']
        if disease_name in exclude:
          continue
        disease_index = dis2int[disease_name]

        for doc_elem in disease_elem:
          id = doc_elem.attrib['id']
          if not id in doc2labels:
            doc2labels[id] = [0] * len(dis2int)

          disease_label = doc_elem.attrib['judgment']
          doc2labels[id][disease_index] = to_int[disease_label]

  return doc2labels

def get_disease_names(xml, exclude=set()):
  """Get list of diseases from standoff files"""

  disease_names = set()
  tree = et.parse(xml)

  for disease_elem in tree.iter('disease'):
    disease_name = disease_elem.attrib['name']
    if not disease_name in exclude:
      disease_names.add(disease_name)

  return sorted(list(disease_names))

def write_notes_to_files(notes_xml, output_dir):
  """Extract notes from xml and write to files"""

  tree = et.parse(notes_xml)

  for doc in tree.iter('doc'):
    doc_id = doc.attrib['id']
    doc_text = doc[0].text

    file_name = os.path.join(base, outdir, '%s.txt' % doc_id)
    out_file = open(file_name, 'w')
    out_file.write(doc_text)

if __name__ == "__main__":

  base = '/Users/Dima/Loyola/Data/'
  notes = 'Comorbidity/Xml/obesity_patient_records_test.xml'
  xml = 'Comorbidity/Xml/obesity_standoff_annotations_test.xml'
  outdir = 'Comorbidity/Text/Test/'

  annot_xml = os.path.join(base, xml)

  doc2labels = parse_standoff_vectorized(annot_xml, 'intuitive')
  print doc2labels
