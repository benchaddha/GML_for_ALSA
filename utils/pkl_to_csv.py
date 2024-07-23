# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# pkl_to_csv.py
#
# Identification: /pkl_to_csv.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//
import pickle as pkl
import pandas as pd
# with open("data/all_global_parse_fg.pkl", "rb") as f:
#     object = pkl.load(f)
    
# df = pd.DataFrame(object)
# df.to_csv(r'file.csv')
import pickletools
with open("data/all_global_parse_fg.pkl", "rb") as f:
    pickletools.dis(f)
