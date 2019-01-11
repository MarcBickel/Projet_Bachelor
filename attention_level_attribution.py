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
from cross_validation import *
from validation import *
from final_output import *
from interpolation import *

list_of_ids = set([ 71,  74,  75,  76,  77,  78,  79,  81,  83,  84,  85,  88,  89,
                        90,  93,  94,  95,  98, 101, 103, 104, 105, 106, 107, 107, 109,
                        111, 112, 115, 116])

#création du dict (données préparées à la main sur une frame du début)
pos_ids_lesson03 = [
    {#cam01
        '109' : [11.615488 , 175.00432],
        '77' : [255.46295 , 192.63771],
        '107' : [590.8298 , 125.969315], 
        '76' : [734.08356 , 125.74278], 
        '75': [1123.1455 , 126.00644], 
        '111' : [170.54427 , 299.97433],
        '84' : [456.7496 , 237.79659],
        '94': [806.05273 , 233.13977],
        '101': [998.19214 , 228.8077],
        '104': [1230.5897 , 224.23285],
        '106': [1338.0624 , 192.56294],
        '112' : [1512.5647 , 210.68916],
        '78' : [11.560406 , 443.2507],
        '115' : [412.18192 , 402.96262],
        '79': [1006.9935 , 340.79202],
        '71': [1315.766 , 322.5751]
    }, 
    {#cam02 
        '116' : [575, 280],
        '85' : [762.88007 , 192.37157],
        '95' : [680.3075 , 313.04572],
        '90' : [998.45044 , 285.4018],
        '74' : [899.79755 , 505.22498],
        '81' : [1244.8258 , 285.9095], 
        '111' : [1475.2833 , 329.46097], 
        '84' : [1727.6028 , 285.54385], 
        '98' : [1601.4677 , 460.91666], 
        '78' : [1815.5504 , 450.11475], 
        '89' : [115.60209 , 329.2369], 
        '107' : [1639.5898 , 203.30162], 
        '76' : [1727.6028 , 285.54385], 
        '-1' : [1864.6171 , 203.34546], 
        '-1' : [60.742004 , 214.3613]
    },
    {#cam03
        '105' : [47.692318 , 157.77548],
        '89' : [236.3573 , 147.69553],
        '74' : [241.76576 , 283.6714],
        '95' : [519.66504 , 183.8853],
        '88' : [524.86505 , 309.7839],
        '98' : [687.47906 , 320.29364],
        '90' : [692.67145 , 183.86847],
        '85' : [739.88885 , 105.2115],
        '81' : [870.96484 , 189.27504],
        '78' : [902.3105 , 325.56155 ],
        '109' : [ 933.4616 , 126.39253 ],
        '111' : [ 1054.2283 , 231.18103 ],
        '77' : [ 1143.2229 , 146.9791 ],
        '115' : [ 1232.8423 , 330.79755 ],
        '84' : [ 1295.6187 , 205.00574 ],
        '107' : [ 1442.1866 , 110.73251 ],
        '76' : [ 1604.9254 , 141.97223 ],
        '94' : [ 1699.1973 , 230.9881 ],
        '-1' : [ 1835.6898 , 142.23631 ],
        '101' : [ 1882.9062 , 262.5353 ],
        '79' : [ 1914.385 , 393.89328 ]
    }, 
    {#cam04 
        '103' : [ 229.98643 , 198.3532 ],
        '83' : [ 301.8373 , 121.78185 ],
        '105' : [ 531.5874 , 220.90605 ],
        '93' : [ 580.9457 , 333.4659 ],
        '-1' : [ 801.55963 , 130.99716 ],
        '89' : [ 851.18506 , 220.8873 ],
        '116' : [ 1211.2316 , 126.5454 ],
        '-1' : [ 1242.6442 , 54.288193 ],
        '95' : [ 1368.5049 , 239.10063 ],
        '85' : [ 1453.8611 , 126.73397 ],
        '74' : [ 1643.1106 , 450.723 ]
    }
]

def compute_and_append_ys(xs, attention_levels, positions_by_ids):
    y_temp = []
    
    for i in range(xs.shape[0]):
        x0, y0 = xs[i][5], xs[i][6] # +1 à cause du numéro de frame

        #create the corresponding ys
        frame_numbers = np.array(xs[:, 1])
        frame_numbers = frame_numbers.astype(int)

        current_id, _ = nearest_id(x0, y0, positions_by_ids)
        first_frame_of_id = first_occurence_of_id(current_id, attention_levels[:, 1])
        y_temp.append(attention_levels[first_frame_of_id + frame_numbers[i]][0])        

    ys = np.array([y_temp])
    ys = ys.astype(int)
    
    return np.concatenate((ys.T, xs), axis=1)


def distance_to_point(xfrom, yfrom, xto, yto):
    '''
    Function that returns the Euclidean distance between two points. Used to later compute the nearest neightbour. 
    '''
    x = xto - xfrom
    y = yto - yfrom
    
    return np.sqrt(x*x + y*y)
    

def nearest_id(x0, y0, positions):
    '''
    Returns the id of the closest point in positions, and the distance to it. 
    
    Arguments
    --------
    x0 -> the x position of the point you want the nearest neightbour of
    y0 -> the y position of the point you want the nearest neightbour of
    positions -> dict containing ids as keys and tuples of coordinates as values
    
    '''
    nearest = 'No answer was found'
    distance = sys.maxsize
    for key, coordinates in positions.items():
        new_dist = distance_to_point(x0, y0, coordinates[0], coordinates[1])
        if (new_dist < distance):
            nearest = key
            distance = new_dist
            
    return int(nearest), distance

def first_occurence_of_id(id_number, array):
    '''
    Returns the first occurence of the `id_number` in the numpy `array`.
    '''
    indices = np.where(array==(id_number))
    if (indices[0].size != 0):
        minim = np.min(indices)
        return minim
    else:
        return 0


#multiple cameras
def put_id_and_compute_and_append_ys(xs, attention_levels, coordinates_by_ids):
    y_temp = []
    ids = []
    
    cam_numbers = xs[:, 0]
    xs = xs[:, 1:]
    
    for i in range(xs.shape[0]):
        x = xs[i]
        cam_number = cam_numbers[i]
        
        x0, y0 = x[4], x[5] # +1 à cause du numéro de frame, + à cause du numéro de caméra ? (pas encore fait)

        positions_by_ids = coordinates_by_ids[int(cam_number) - 1]
        
        #create the corresponding ys
        frame_numbers = np.array(xs[:, 0])
        frame_numbers = frame_numbers.astype(int)

        current_id, _ = nearest_id(x0, y0, positions_by_ids)
        ids.append(current_id)
        first_occurence = first_occurence_of_id(int(current_id), attention_levels[:, 1])
        level = first_occurence + frame_numbers[i]
        #print(current_id, first_occurence, frame_numbers[i], level)
        y_temp.append(attention_levels[level][0])        

    ys = np.array(y_temp)
    ys = ys.astype(int)
    ids = np.array(ids)
    
    result_temp = np.concatenate((ids[:, np.newaxis], ys[:, np.newaxis], xs[:, 1:]), axis=1)#enlever le numéro des caméras pour ne pas breaker le reste du code
    #enlever les ids de -1, ce sont les gens qui n'ont pas participé
    result = []
    
    for i in range(result_temp.shape[0]): 
        if (result_temp[i][0] != -1): 
            result.append(result_temp[i])
    
    return np.array(result)