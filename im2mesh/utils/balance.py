import os 
import random
import numpy as np


data_dir = 'data/ShapeNet'

import glob

files = glob.iglob(f"{data_dir}/**/train.lst", recursive=True)
counts = {f: sum(1 for line in open(f)) for f in files}
print(counts)

t = max(list(counts.values()))
for fl in counts.keys():
    fc = counts[fl]
    if fc >= t:
        continue
    c = random.choices(range(fc), k=t-fc)
    with open(fl, 'r+') as f:
        ln = f.readlines()
        ln = [l.replace('\n', '') for l in ln]
        ln = [ln[i] for i in c]
        f.write("\n")
        f.write("\n".join(ln))

    with open('ap_logs.lst', 'a+') as f:
        f.write("\n")
        f.write("\n".join(ln))

files = glob.iglob(f"{data_dir}/**/train.lst", recursive=True)
counts = {f: sum(1 for line in open(f)) for f in files}
print(counts)
