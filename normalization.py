import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import sys
import numpy as np
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt

from attention_level_attribution import *
from cross_validation import *
from validation import *
from final_output import *
from interpolation import *

def find_size_cloud_np(matrix):
    temp = np.copy(matrix)
    keypoints_x = temp[::2]
    keypoints_y = temp[1::2]
    
    x_min = np.min(keypoints_x[np.nonzero(keypoints_x)]) 
    x_max = np.max(keypoints_x[np.nonzero(keypoints_x)])
    y_min = np.min(keypoints_y[np.nonzero(keypoints_y)])
    y_max = np.max(keypoints_y[np.nonzero(keypoints_y)])
    
    return x_min, x_max, y_min, y_max

def find_size_cloud(keypoints):
    temp = np.copy(keypoints)
    x_min, x_max, y_min, y_max = 100000, -1, 100000, -1
    
    for i in range(temp.shape[0]):
        t = temp[i]
        
        if (t != 0): #ignorer coordonnées égales à 0
            if (i % 2 == 0): #indice pair, donc coordonnée x
                if (t > x_max):
                    x_max = t
                elif (t < x_min):
                    x_min = t

            else: #indice impair, donc coordonnée y
                if (t > y_max):
                    y_max = t
                elif (t < y_min):
                    y_min = t
                    
    return x_min, x_max, y_min, y_max

def center_keypoints(keypoints):
    result = np.copy(keypoints)
    x_min, x_max, y_min, y_max = find_size_cloud(result)
    
    x_mean = (x_max - x_min)/2 + x_min
    y_mean = (y_max - y_min)/2 + y_min
    
    keypoints_x = result[::2]
    keypoints_y = result[1::2]
    
    for i in range(result.shape[0]):
        t = result[i]
        if (t != 0): #ignorer coordonnées égales à 0
            if (i % 2 == 0): #indice pair, donc coordonnée x
                result[i] = t - x_mean

            else: #indice impair, donc coordonnée y
                result[i] = t - y_mean
    return result
    
    
def resize_keypoints(matrix):
    result = np.copy(matrix)
    x_min, x_max, y_min, y_max = find_size_cloud(result)
    
    factor = 1 / np.max([x_max - x_min, y_max - y_min])
    
    result = np.multiply(result, factor)
    
    return result

def delete_confidence(matrix):
    result = np.copy(matrix)
    if (matrix.shape[0] == 75):
        return np.delete(result, np.arange(2, result.size, 3))
    else:
        raise TypeError('The matrix had shape ' + str(matrix.shape) + ' which was not expected. ')

def delete_legs(matrix):
    temp = np.copy(matrix)
    
    #places des coordonnées des jambes
    ids_legs = [9, 8, 12, 10, 13, 11, 24, 21, 14, 23, 22, 19, 20]
    
    for i in ids_legs:
        temp[i*2] = 0
        temp[i*2+1] = 0
        
    return temp


def normalize(matrix):
    temp = np.copy(matrix)
    result = []
    for i in range(temp.shape[0]):
        row = temp[i, 2:]
        row_norm = center_keypoints(resize_keypoints(delete_legs(delete_confidence(row))))
        
        result.append(np.append(temp[i, :3], row_norm))
    
    return np.array(result)