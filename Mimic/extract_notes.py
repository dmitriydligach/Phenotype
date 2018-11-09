#! /usr/bin/env python3
import pandas, string, os, math

NOTES_CSV = '/Users/Dima/Loyola/Data/MimicIII/Source/NOTEEVENTS.csv'
OUT_DIR = '/Users/Dima/Loyola/Data/MimicIII/Admissions/Text/'

def write_admissions_to_files():
  """Each admission is written to a separate file"""

  frame = pandas.read_csv(NOTES_CSV, dtype='str')

  for rowid, hadmid, text in zip(frame.ROW_ID, frame.HADM_ID, frame.TEXT):
    if pandas.isnull(hadmid):
      print('empty hadmid for rowid', rowid)
    else:
      printable = ''.join(c for c in text if c in string.printable)
      outfile = open('%s%s.txt' % (OUT_DIR, hadmid), 'a')
      outfile.write(printable + '\n')
      outfile.write('\n**************************\n\n')

def write_patients_to_files():
  """Write files to one directory. Group by patient."""

  frame = pandas.read_csv(NOTES_CSV)

  for row_id, subj_id, text in zip(frame.ROW_ID, frame.SUBJECT_ID, frame.TEXT):
    printable = ''.join(c for c in text if c in string.printable)
    outfile = open('%s%s.txt' % (OUT_DIR, subj_id), 'a')
    outfile.write(printable + '\n')

if __name__ == "__main__":

  write_admissions_to_files()
