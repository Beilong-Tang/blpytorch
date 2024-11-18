###
### This script tests if librispeech dynamic dataset and libri2mix target dataset works correct.
###

import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from data.target_dataset import TargetDMDataset, TargetDataset
from data.blind_dataset import BlindDataset
from utils.io import make_path

### 1. Test DM Dataset ###
### config ###
# SCP_PATH = "/public/home/qinxy/zbang/data/LibriSpeech/scp/train/train_clean_100_360.pt"
# OUTPUT_DIR = make_path(
#     "/public/home/qinxy/bltang/ml_framework_slurm/data/output/librispeech_dm",
#     is_dir=True,
# )
# NUM = 20
# ##############
# print("testing dm dadataset")
# dataset = TargetDMDataset(SCP_PATH, -1)
# dataset.test(OUTPUT_DIR, NUM)
# print("done")

### 2. Test Normal Dataset ###
### config ###
# print("testing normal dataset")
# CLEAN_PATH = (
#     "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/s1.scp"
# )

# REGI_PATH = "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/bltang/regi.scp"

# MIX_PATH = (
#     "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/mix.scp"
# )

# OUTPUT_DIR = make_path(
#     "/public/home/qinxy/bltang/ml_framework_slurm/data/output/libri2mix_test",
#     is_dir=True,
# )
# NUM = 20
# ##############

# dataset = TargetDataset(MIX_PATH, REGI_PATH, CLEAN_PATH, -1)
# dataset.test(OUTPUT_DIR, NUM)


### 3. Test libri2mix blind dataset ###
### config ###
print("testing blind normal dataset")
S1_PATH = "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/s1.scp"

S2_PATH = "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/s2.scp"

MIX_PATH = "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/train/all/mix.scp"

OUTPUT_DIR = make_path(
    "/public/home/qinxy/bltang/ml_framework_slurm/data/output/libri2mix_blind_test",
    is_dir=True,
)
NUM = 20
##############

dataset = BlindDataset(MIX_PATH, S1_PATH, S2_PATH, -1)
dataset.test(OUTPUT_DIR, NUM)
