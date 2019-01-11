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
from validation import *
from final_output import *
from interpolation import *

def train_NNmodel_on_GPU(model, keypoints_train, attention_levels, positions_by_ids, learning_rate, loss_fn, number_of_epochs):

    #to run on GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in log_progress(range(number_of_epochs), name='Epoch'):
        for i in range(keypoints_train.shape[0]):
            #élimine le numéro de frame (non utile au NN)
            current_keypoints = keypoints_train[i][1:]

            x0 = current_keypoints[4]
            y0 = current_keypoints[5]

            current_id, _ = nearest_id(x0, y0, positions_by_ids)
            frame_number = int(keypoints_train[i][0])
            first_frame_of_id = first_occurence_of_id(current_id, attention_levels[:, 1])

            #transforme en data compréhensible par le NN
            x = torch.tensor(current_keypoints, dtype = torch.float)
            y = torch.tensor([attention_levels[first_frame_of_id + frame_number][0]], dtype=torch.float)

            x.to(device)
            y.to(device)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            #print('Epoch: ', epoch, ' Column: ', i, ' Loss: ', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


























def compute_accuracy(y_pred, y, threshold):
    if (y - y_pred < threshold and y_pred - y < threshold):
        return 1
    else:
        return 0

def get_accuracy(model, matrix_x, matrix_y, dict_pos_by_ids, accuracy_threshold):
    ys = []
    y_preds = []
    accuracies = []
    losses = []
    
    loss_function = torch.nn.L1Loss()
    
    with torch.no_grad():
        for i in range(matrix_x.shape[0]):
            #élimine le numéro de frame
            current_keypoints = matrix_x[i][1:]

            x0 = current_keypoints[4] 
            y0 = current_keypoints[5]

            current_id, _ = nearest_id(x0, y0, dict_pos_by_ids)
            frame_number = int(matrix_x[i][0])
            first_frame_of_id = first_occurence_of_id(current_id, matrix_y[:, 1])

            #transforme en data compréhensible par le NN
            x = torch.tensor(current_keypoints, dtype = torch.float)
            y = torch.tensor([matrix_y[first_frame_of_id + frame_number][0]], dtype=torch.float)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)

            loss = loss_function(y_pred, y)
            accuracy = compute_accuracy(y_pred, y, accuracy_threshold)

            ys.append(y.item())
            y_preds.append(np.round(y_pred.item(), 3))
            accuracies.append(np.round(accuracy, 3))
            losses.append(np.round(loss.item(), 3))
    
    
    
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)
    #print('Mean loss: ', mean_loss)
    #print('Mean accuracy: ', mean_accuracy)
    
    data = np.array([ys, y_preds, losses, accuracies])
    headers = ['Expected y', 'Predicted y', 'Loss', 'Accuracy']
    return pd.DataFrame(data.T, columns=headers), mean_loss, mean_accuracy





























import copy

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_with_train(model, x, y, k_indices, k, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold):
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    m_train = x[tr_indice]
    m_test = x[te_indice]
    
    train_NNmodel_on_GPU(model, m_train, y, positions_by_ids, learning_rate, loss_fn, epochs)
    return get_accuracy(model, m_test, y, positions_by_ids, accuracy_threshold)

def run_cross_validation_with_train(model, x, y, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold, k_fold=5, seed=1):
    k_indices = build_k_indices(x, k_fold, seed)
    results, losses, accuracies = pd.DataFrame(), [], []
    for k in range(k_fold):
        m2 = copy.deepcopy(model)
        results_temp, loss_temp, acc_temp = cross_validation_with_train(m2, x, y, k_indices, k, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold)
        results = results.append(results_temp)
        losses.append(loss_temp)
        accuracies.append(acc_temp)
        print("Accuracy on sample " + str(k+1) + ": " + str(acc_temp) + " with a mean loss of:" + str(loss_temp))
        
    return results, pd.DataFrame(losses, columns=['Mean Losses']), pd.DataFrame(accuracies, columns=['Mean accuracies'])



































def train_NNmodel_and_cross_validation(model, keypoints_train, keypoints_validation, attention_levels, positions_by_ids, learning_rate, loss_fn, number_of_epochs, accuracy_threshold=0.5):
    #to run on GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    results, losses, accuracies = pd.DataFrame(), [], []

    for epoch in log_progress(range(number_of_epochs), name='Epoch'):
        for i in range(keypoints_train.shape[0]):
            #élimine le numéro de frame (non utile au NN)
            current_keypoints = keypoints_train[i][1:]

            x0 = current_keypoints[4]
            y0 = current_keypoints[5]

            current_id, _ = nearest_id(x0, y0, positions_by_ids)
            frame_number = int(keypoints_train[i][0])
            first_frame_of_id = first_occurence_of_id(current_id, attention_levels[:, 1])

            #transforme en data compréhensible par le NN
            x = torch.tensor(current_keypoints, dtype = torch.float)
            y = torch.tensor([attention_levels[first_frame_of_id + frame_number][0]], dtype=torch.float)

            x.to(device)
            y.to(device)

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            #print('Epoch: ', epoch, ' Column: ', i, ' Loss: ', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        results_temp, loss_temp, acc_temp = get_accuracy(model, keypoints_validation, attention_levels, positions_by_ids, accuracy_threshold)
        results = results.append(results_temp)
        losses.append(loss_temp)
        accuracies.append(acc_temp)
            
    return results, losses, accuracies


def cross_validation_per_epoch(model, x, y, k_indices, k, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold):
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    m_train = x[tr_indice]
    m_test = x[te_indice]
    
    return train_NNmodel_and_cross_validation(model, m_train, m_test, y, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold)


def run_cross_validation_per_epoch(model, x, y, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold, k_fold=5, seed=1):
    k_indices = build_k_indices(x, k_fold, seed)
    results, losses, accuracies = pd.DataFrame(), [], []
    for k in log_progress(range(k_fold), name='Kth-fold'):
        m2 = copy.deepcopy(model)
        results_temp, loss_temp, acc_temp = cross_validation_per_epoch(m2, x, y, k_indices, k, positions_by_ids, learning_rate, loss_fn, epochs, accuracy_threshold)
        results = results.append(results_temp)
        losses.append(loss_temp)
        accuracies.append(acc_temp)
        
    return results, pd.DataFrame(losses), pd.DataFrame(accuracies)


































def cross_validation_visualization(lambdas, loss_train, loss_test):
    """visualization the curves of loss_train and loss_test."""
    plt.semilogx(lambdas, loss_train, marker=".", color='b', label='train error')
    plt.semilogx(lambdas, loss_test, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_mse")
    
def cross_validation_visualization_accuracy(epochs, accs, save=False, filename="cross_validation_acc"):
    """visualization the curve of accuracy"""
    plt.plot(epochs, accs, marker=".", color='r', label='accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    if (save):
        plt.savefig(filename)
    
def cross_validation_visualization_accuracy_multiple(epochs, accs, save=False, filename="cross_validation_acc_multiple"):
    """visualization the curve of accuracy"""
    
    for i in range(accs.shape[0]):
        plt.plot(epochs, accs[i], marker=".", color='r', label=str(i+1)+'th accuracy')
        
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    if (save):
        plt.savefig(filename)
    
def cross_validation_visualization_mean(epochs, means, save=False, filename = "cross_validation_mean"):
    """visualization the curve of means"""
    plt.plot(epochs, means, marker=".", color='b', label='means')
    plt.xlabel("epoch")
    plt.ylabel("mean loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    if (save):
        plt.savefig(filename)
    
def cross_validation_visualization_mean_multiple(epochs, means, save=False, filename='cross_validation_mean_multiple'):
    """visualization the curve of mean error"""
    
    for i in range(means.shape[0]):
        plt.plot(epochs, means[i], marker=".", color='b', label=str(i+1) + 'th means')
        
    plt.xlabel("epoch")
    plt.ylabel("mean error")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    if (save):
        plt.savefig(filename)
        































def train_NNmodel_batch(model, keypoints_train, attention_levels, positions_by_ids, learning_rate, loss_fn, number_of_epochs, batch_size):

    #to run on GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in log_progress(range(number_of_epochs), name='Epoch'):
        
        #get number of random indexes equal to batch_size
        indexes_batch = np.random.choice(keypoints_train.shape[0], batch_size, replace=False)
        #get the columns that correspond to these 
        x_with_frames = keypoints_train[indexes_batch]

        #create a matrix that is (batch_size * D_in) dimensions
        x_temp = x_with_frames[:, 1:]

        #give directly the matrix as input to the model
        x = torch.tensor(x_temp, dtype = torch.float, device=device)

        #create the corresponding ys
        frame_numbers = np.array(x_with_frames[:, 1])
        frame_numbers = frame_numbers.astype(int)
        y_temp = []
        for i in range(frame_numbers.shape[0]):
            x0, y0 = x_temp[i][4], x_temp[i][5]
            current_id, _ = nearest_id(x0, y0, positions_by_ids)
            first_frame_of_id = first_occurence_of_id(current_id, attention_levels[:, 1])
            y_temp.append(attention_levels[first_frame_of_id + frame_numbers[i]][0])           
        
        
        y = torch.tensor([y_temp], dtype=torch.float, device = device)
        y = y.transpose_(0, 1)
        
        
        
        #should speed up considerably the training, and maybe better the results
        #DELETE THE FOR LOOP AND ADAPT CODE
        
        #for i in range(keypoints_train.shape[0]):
        #    #élimine le numéro de frame (non utile au NN)
        #    current_keypoints = keypoints_train[i][1:]

        #    x0 = current_keypoints[4]
        #    y0 = current_keypoints[5]

        #    current_id, _ = nearest_id(x0, y0, positions_by_ids)
        #    frame_number = int(keypoints_train[i][0])
        #    first_frame_of_id = first_occurence_of_id(current_id, attention_levels[:, 1])

            #transforme en data compréhensible par le NN
        #    x = torch.tensor(current_keypoints, dtype = torch.float, device=device)
        #    y = torch.tensor([attention_levels[first_frame_of_id + frame_number][0]], dtype=torch.float, device=device)
        
        

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        #print('Epoch: ', epoch, ' Column: ', i, ' Loss: ', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



























#validation for batch version of NN































def train_NNmodel_batch_precomputed(model, matrix, learning_rate, loss_fn, number_of_epochs, batch_size):

    #to run on GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in log_progress(range(number_of_epochs), name='Epoch'):
        
        ys = matrix[:, 0] #prend les niveaux d'attention 
        xs = matrix[:, 1:]
        
        #get number of random indexes equal to batch_size
        indexes_batch = np.random.choice(xs.shape[0], batch_size, replace=False)
        #get the rows that correspond to these 
        x_with_frames = xs[indexes_batch]

        #create a matrix that is (batch_size * D_in) dimensions
        x_temp = x_with_frames[:, 1:]

        #give directly the matrix as input to the model
        x = torch.tensor(x_temp, dtype = torch.float, device=device)         
        
        y_temp = ys[indexes_batch]
        y = torch.tensor([y_temp], dtype=torch.float, device = device)
        y = y.transpose_(0, 1)
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        #print('Epoch: ', epoch, ' Column: ', i, ' Loss: ', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


















#validation for batch version of NN, with precomputed ys
        





def validation_batch(model, x, k_indices, k, learning_rate, loss_fn, epochs, batch_size, accuracy_threshold):
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    m_train = x[tr_indice]
    m_test = x[te_indice]
    
    return train_NNmodel_validation_batch(model, m_train, m_test, learning_rate, loss_fn, epochs, batch_size, accuracy_threshold)


def run_validation_batch(model, x, learning_rate, loss_fn, epochs, batch_size=64, accuracy_threshold=0.5, k_fold=5, seed=1):
    k_indices = build_k_indices(x, k_fold, seed)
    results, losses, accuracies = [], [], []
    for k in tqdm_notebook(range(k_fold), desc='Kth-fold'):
        m2 = copy.deepcopy(model)
        results_temp, loss_temp, acc_temp = validation_batch(m2, x, k_indices, k, learning_rate, loss_fn, epochs, batch_size, accuracy_threshold)
        results.append(results_temp)
        losses.append(loss_temp)
        accuracies.append(acc_temp)
        
    return pd.DataFrame(results), pd.DataFrame(losses), pd.DataFrame(accuracies)












def compute_accuracy_vector(y_pred, y, threshold):
    count = np.count_nonzero(np.logical_and((y - y_pred) < threshold, (y_pred - y) < threshold))
    return count / y_pred.size
        

def get_accuracy_vector(model, matrix, batch_size, accuracy_threshold):
    matrix_selected = matrix
    
    ys = matrix_selected[:, 0]
    xs = matrix_selected[:, 2:]
    
    y_preds = np.empty([0, 1])
    accuracies = []
    losses = []
    
    loss_function = torch.nn.L1Loss()
    
    with torch.no_grad():

        #transforme en data compréhensible par le NN et élimine le numéro de frame
        x = torch.tensor(xs, dtype = torch.float)
        y = torch.tensor([ys], dtype=torch.float)
        y.transpose_(0, 1)
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        loss = loss_function(y_pred, y)
        accuracy = compute_accuracy_vector(y_pred.cpu().numpy(), y.cpu().numpy(), accuracy_threshold)

        y_preds = np.vstack((y_preds, np.round(y_pred, 3)))
        accuracies.append(np.round(accuracy, 3))
        losses.append(np.round(loss.item(), 3))
    
    
    
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)
    
    data = np.array([ys, y_preds, losses, accuracies])
    #headers = ['Expected y', 'Predicted y', 'Loss', 'Accuracy']
    return data, mean_loss, mean_accuracy



def train_NNmodel_validation_batch(model, matrix_train, matrix_validation, learning_rate, loss_fn, number_of_epochs, batch_size, accuracy_threshold):
    
    results, losses, accuracies = [], [], []
   
    
    #to run on GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm_notebook(range(number_of_epochs), desc='Epoch', leave=False):
        
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
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        results_temp, loss_temp, acc_temp = get_accuracy_vector(model, matrix_validation, batch_size, accuracy_threshold)
        results.append(results_temp)
        losses.append(loss_temp)
        accuracies.append(acc_temp)
            
    return results, losses, accuracies

def run_new_validation_batch(model, m_train, m_validation, learning_rate, loss_fn, epochs, batch_size=64, accuracy_threshold=0.5, k_fold=5):
    results, losses, accuracies = [], [], []
    for k in tqdm_notebook(range(k_fold), desc='Kth-fold'):
        m2 = copy.deepcopy(model)
        results_temp, loss_temp, acc_temp = train_NNmodel_validation_batch(model, m_train, m_validation, learning_rate, loss_fn, epochs, batch_size, accuracy_threshold)
        results.append(results_temp)
        losses.append(loss_temp)
        accuracies.append(acc_temp)
        
    return results, losses, accuracies