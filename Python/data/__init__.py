import csv
import os
import sys


csv_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(csv_limit)
        break
    except OverflowError:
        csv_limit = int(csv_limit / 10)


PARENT_PATH = os.path.abspath("../..")
DATA_DIR = os.path.join(PARENT_PATH, "data")
SUBMISSIONS_DIR = os.path.join(PARENT_PATH, "submissions")
