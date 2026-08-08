"""Shared utilities."""

import os
import pickle
import random

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_sample_stage(path):
    with open(os.path.join(path, "sample_stage.pkl"), "rb") as f:
        return pickle.load(f)
