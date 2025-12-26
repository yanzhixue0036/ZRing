import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import random
import time

from Trainres.MLP import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def compute_aare(predictions, targets):
    predictions_original = torch.expm1(predictions)
    targets_original = torch.expm1(targets)

    relative_errors = torch.abs(predictions_original - targets_original) / torch.abs(targets_original)
    aare = torch.mean(relative_errors)
    return aare.item()

def extract_features(row, num = 32):
    non_one_indices = [i for i in range(len(row) - 1, -1, -1) if row[i] != 1]
    if len(non_one_indices) == 0:
        return [0] + [1] * num

    first_non_one_index = non_one_indices[0]
    left_values = row[max(0, first_non_one_index - num + 1):first_non_one_index + 1][::-1]
    left_values = np.pad(left_values, (0, max(0, num - len(left_values))), constant_values=1)
    if(not len(left_values) == num): print(left_values)
    return [first_non_one_index] + left_values.tolist()

def load_and_filter_data(file_path, r_size = 6, truncation_num = 32):
    data = np.load(file_path)
    
    X = np.array([row[:pow(2, r_size) + 2] for row in data])
    # X = np.array([extract_features(row[:pow(2, r_size)], num = truncation_num) for row in data])
    y = np.array([row[pow(2, r_size) + 2] for row in data])

    def filt(pre_X, pre_y):
        filtered_X = []
        filtered_y = []
        for xi, yi in zip(pre_X, pre_y):
            if yi < 1 or yi > 10e13:
                if np.random.rand() <= 0:
                    filtered_X.append(xi)
                    filtered_y.append(yi)
            else:
                filtered_X.append(xi)
                filtered_y.append(yi)
        return filtered_X, filtered_y

    X, y = filt(X, y)
    X = torch.from_numpy(np.array(X, dtype=np.float32))
    y = torch.from_numpy(np.array(y, dtype=np.float32))
    return X, y

def analyze_distribution(y):
    log_y = np.log10(y.numpy())
    bins = np.arange(np.floor(log_y.min()), np.ceil(log_y.max()) + 1, 1)
    hist, edges = np.histogram(log_y, bins=bins)

    for i in range(len(hist)):
        print(f"Range {edges[i]:.1f} to {edges[i+1]:.1f}: {hist[i]}")

def test_model(model, data_loader):
    model.eval() 
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            predictions = model(batch_X).squeeze()
            all_predictions.append(predictions)
            all_targets.append(batch_y)
            
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    aare = compute_aare(all_predictions, all_targets)
    return aare

if __name__ == "__main__":
    data_file = "ZRingset/test.npy"
    print(data_file)

    r_size = 8
    truncation_num = 16
    model_file = f"Trainres/l_1=5,l_2=3,Nx=8.pth"
    X, y = load_and_filter_data(data_file, r_size = r_size, truncation_num = truncation_num)

    analyze_distribution(y)

    y = torch.log1p(y) 

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False) 

    model = MLP(input_dim = pow(2, r_size) + 2)

    model.load_state_dict(torch.load(model_file))

    t1 = time.time()
    aare = test_model(model, data_loader)
    t2 = time.time()
    print(f"AARE on {data_file}: {aare:.4f}")
    print(f"test time: {t2 - t1}")
