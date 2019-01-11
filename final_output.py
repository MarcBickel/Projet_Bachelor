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
from interpolation import *

#================================================================================================#
#Imports and preparation for OpenPose

import sys
import cv2
import os
from sys import platform
from matplotlib import pyplot as plt


__file__ = os.path.realpath('C:/Users/Marku/Documents_no_sync/Projet_Bachelor/openpose/build/python/openpose/__init__.py')
sys.path.append(os.path.realpath('C:\\Users\\Marku\\Documents_no_sync\\Projet_Bachelor\\openpose\\build\\examples\\tutorial_api_python'))
dir_path = os.path.dirname(os.path.realpath(__file__))
if platform == "win32": sys.path.append(dir_path + '/../../python/openpose/');
else: sys.path.append('../../python');

try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "368x-1"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

print("Imports and start of openpose success !")
#================================================================================================#

#permet de stocker le NN entraîné
def train_NN_def(): 
    '''
    Trains the model used for the results presented in the report. Hyper-parameters are already set, but you can change them if you like. 
    '''
    #DATA LOAD

    ## PARAMETERS

    keypoints_lesson03_file = 'keypoints_pierre03_augmented'

    #liste de tous les ids dans toute l'expérience
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

    learning_rate = 1e-7
    number_of_epochs = 100000
    loss_fn = torch.nn.MSELoss()
    batch_size = 512

    #excel_file = 'C:/Users/Marku/Documents_no_sync/Projet_Bachelor/data/C2_alle.xlsx'
    excel_file_colab = './data/C2_alle.xlsx'

    # Load spreadsheet
    xl = pd.ExcelFile(excel_file_colab)

    # Load a sheet into a DataFrame by name: df1
    df1 = xl.parse('C2_alle')

    print('Successfully imported the excel file')



    ############################################### END OF PARAMETERS

    keypoints_cam1 = np.load(keypoints_lesson03_file + '.npy').T
    print('The matrix containing all the keypoints has shape ' + str(keypoints_cam1.shape))

    #df1 = df1[(df1['Lesson'] == 0)]

    df_cam1 = df1[['Global Attention Rating', 'Participant']]

    df_cam_plus = attention_levels_augmentation(df_cam1, list_of_ids)

    print('linear interpolation done')

    plt.rcParams["figure.figsize"] = [15, 10]

    matrix_cam1 = put_id_and_compute_and_append_ys(keypoints_cam1, df_cam_plus, pos_ids_lesson03)

    print('appended ids and ys')
    print(matrix_cam1.shape)

    matrix_cam1_norm = normalize(matrix_cam1)

    print('normalized data')


    # D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H, D_out = matrix_cam1_norm.shape[1] - 2, 1500, 1 
    #la première ligne qui contient le numéro de frame ne nous intéresse pas dans le NN


    
    network = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.BatchNorm1d(H),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(), 
                torch.nn.BatchNorm1d(H),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_out)
            )
    
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    matrix_train = matrix_cam1_norm

    for epoch in tqdm_notebook(range(number_of_epochs), desc='Epoch', leave=False, position=0, mininterval=1):
        
        #get number of random indexes equal to batch_size
        indexes_batch = np.random.choice(matrix_train.shape[0], batch_size, replace=False)
        #get the rows that correspond to these 
        matrix_selected = matrix_train[indexes_batch]
        
        ys = matrix_selected[:, 0] #prend les niveaux d'attention (les y)
        xs = matrix_selected[:, 2:] #create a matrix that is (batch_size * D_in) dimensions / ignores the frame numbers

        #give directly the matrix as input to the model
        x = torch.tensor(xs, dtype = torch.float, device=device)         
        
        y = torch.tensor([ys], dtype=torch.float, device = device)
        y = y.transpose_(0, 1)
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = network(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    
    return network
    

def create_output_image(path_input, path_output, coordinates, y_predict):
    '''
    Creates the image with background input image, and attention levels printed in red in foreground. 
    '''
    
    HEAD_SIZE = 50
    
    fnt = ImageFont.truetype('C://WINDOWS/Fonts/Arial.ttf', 100)
    img = Image.open(path_input)
    d = ImageDraw.Draw(img)
    
    
    for i in range(len(y_predict)):
        head_coord = coordinates[i]
        d.text((head_coord[0] - HEAD_SIZE / 2, head_coord[1] - 3 * HEAD_SIZE), str(y_predict[i]), font=fnt, fill=(255,0,0))
        d.rectangle([(head_coord[0] - HEAD_SIZE, head_coord[1] - HEAD_SIZE),
                     (head_coord[0] + HEAD_SIZE, head_coord[1] + HEAD_SIZE)],
                    outline=(255,0,0), width = 5)
    

    img.save(path_output)
    img.show()

    
#prédit les labels correspondants à une image
def predict_from_image(path_input, path_output, model):
    '''
    Runs a model on an input image, and outputs a graphical result. 
    Parameters:
    @path_input: the image where the attention levels have to be computed
    @path_output: the name of where you want the results to be stored
    @model: the Neural Network used to predict the student attention levels
    '''
    with torch.no_grad():
        
        img = cv2.imread(path_input)
        keypoints = openpose.forward(img)
        
        #vérifier la shape et l'orientation de la matrice coordinates
        coordinates = []
        
        for j in range(keypoints.shape[0]):
            coordinates.append(np.hstack((np.array([0, 0]), keypoints[j].flatten())))
        
        coordinates = np.array(coordinates)

        coord_norm = normalize(coordinates)
        coord_norm = coord_norm[:, 2:]
        coordinates = coordinates[:, 2:]

        
        x = torch.tensor(coord_norm, dtype = torch.float)

        y_pred = model(x)
        
        ys = y_pred.cpu().numpy()
        ys = ys.astype(int).flatten()
            
        #changer la forme, mais l'idée est de prendre les coordonnées du centre des têtes uniquement (place 0)
        coordinates_heads = [] 
        
        for i in range(coordinates.shape[0]):
            coordinates_heads.append([coordinates[i, 0], coordinates[i, 1]])
    
    
    return create_output_image(path_input, path_output, coordinates_heads, ys)
    
