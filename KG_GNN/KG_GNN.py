import KGGNN
from scipy import sparse
import numpy as np
import torch
import rabbit

def nnz_extract(RowPtr, ColIdx):
