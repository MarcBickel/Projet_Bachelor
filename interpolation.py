import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import sys
import numpy as np
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from normalization import *
from attention_level_attribution import *
from cross_validation import *
from validation import *
from final_output import *


#For the attention levels

def separate_by_ids(matrix, ids_list):
    
    by_id = []
    
    #sort by ids
    for id in ids_list:
        attentions = matrix[(matrix['Participant'] == id)].values
        
        if (len(attentions) != 0):
            by_id.append(attentions)
        
    return by_id
    

def interpolate(row):
    '''
    Take in an array of attention levels for a single id. '''
    result = []
    
    for i in range(row.shape[0] - 1):
        result.append(row[i])
        inter = (row[i] + row[i+1]) / 2
        result.append(inter)
    
    result.append(row[-1]) #last element of the row
    
    return np.array(result)


    
def recombine_after_interpolation(list_of_rows):
    
    return np.array([item for sublist in list_of_rows for item in sublist])
    

def attention_levels_augmentation(matrix, ids_list):
    temp = separate_by_ids(matrix, ids_list)
    
    list_rows = []
    
    for r in temp:
        t = interpolate(r)
        list_rows.append(t)
    
    augmented = recombine_after_interpolation(list_rows)
    return augmented
    
    
#For the coordinates: simply extract more pictures from the videos, and run openpose as usual on all of them
    