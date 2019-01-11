import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import sys
import numpy as np
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt

from normalization import *
from attention_level_attribution import *
from cross_validation import *
from final_output import *
from interpolation import *

def enlever_ids_validation(matrix, ids):
    training_matrix = []
    validation_matrix = []
    for i in range(matrix.shape[0]):
        row = matrix[i]
        if (row[0] in ids): 
            validation_matrix.append(row[1:])
        else:
            training_matrix.append(row[1:])
            
    return np.asarray(training_matrix), np.asarray(validation_matrix)