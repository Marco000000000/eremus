import os
import mne
import numpy as np
import torch
from utils import config

def get_seed_montage():
    # Load channel locations file
    with open(os.path.join("utils", "SEED_Channel_Location.txt")) as fp:
        lines = [l.split() for l in fp]
    seed_coord = {l[0]: torch.tensor([float(x) for x in l[1:]]) for l in lines}
    # Define SEED electrode names
    seed_channels = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                    'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                    'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']
    # Remove useless channels
    montage = [seed_coord[ch] for ch in seed_channels]
    # Return
    return montage

def get_deap_montage():
    # Get montage
    montage = mne.channels.make_standard_montage('biosemi32')
    # Use DEAP electrodes order
    ch_order = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 14, 15, 12, 29, 
                28, 30, 26, 27, 24, 25, 31, 22, 23, 20, 21, 18, 19, 17, 16]
    # Replace channels
    montage.ch_names = [montage.ch_names[ch] for ch in ch_order]
    montage.dig = montage.dig[:3] + [montage.dig[ch + 3] for ch in ch_order]
    # Return
    return montage

def get_channel_coordinates(num_nodes, dataset="md"):
    if dataset == "md":
        # Get montage
        montage = mne.channels.make_standard_montage('GSN-HydroCel-257')
        # Create coordinate points matrix
        coord_mat = torch.FloatTensor(num_nodes, 3)
        # Fill matrix
        dig = montage.dig[3:-1]
        for i, c in enumerate(config.channels):
            coord_mat[i] = torch.tensor(dig[c]['r'])
        # Meters to centimeters
        coord_mat *= 100
    elif dataset == "seed":
        # Get montage
        montage = get_seed_montage()
        # Create tensor (montage is a list of coordinates)
        coord_mat = torch.stack(montage)
    elif dataset == "deap":
        # Get montage
        montage = get_deap_montage()
        # Create coordinate points matrix
        coord_mat = torch.FloatTensor(num_nodes, 3)
        # Fill matrix
        dig = montage.dig[3:]
        for i in range(num_nodes):
            coord_mat[i] = torch.tensor(dig[i]['r'])
        # Meters to centimeters
        coord_mat *= 100
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    # Return
    return coord_mat

def create_graph(num_nodes, k, dataset="md", delta=5):
    # Get montage
    if dataset == "md":
        montage = mne.channels.make_standard_montage('GSN-HydroCel-257')
    else:
        montage = get_seed_montage()
    # Create coordinate points matrix
    coord_mat = torch.FloatTensor(num_nodes, 3)
    # Fill matrix
    if dataset == "md":
        dig = montage.dig[3:-1]
        for i, c in enumerate(config.channels):
            coord_mat[i] = torch.tensor(dig[c]['r'])
    else:
        dig = montage.dig[3:]
        for i in range(num_nodes):
            coord_mat[i] = torch.tensor(dig[i]['r'])
    # Meters to centimeters
    coord_mat *= 100
    # Initialize tensors
    node_idxs = torch.LongTensor(num_nodes, k)
    node_dists = torch.FloatTensor(num_nodes, k)
    # Compute distances
    for c in range(num_nodes):
        # Current location
        curr_loc = coord_mat[c:c+1]
        # Compute squared distances
        dists = (coord_mat - curr_loc).pow(2).sum(1)
        # Sort distances
        sort_dists, sort_idxs = torch.sort(dists)
        # Add to tensors
        node_idxs[c] = sort_idxs[1:1+k]
        node_dists[c] = sort_dists[1:1+k]
    # Compute edge indices
    node_idxs = node_idxs.view(-1)
    neighbors = torch.arange(num_nodes).repeat_interleave(k)
    edge_index = torch.stack([node_idxs, neighbors], 0)
    # Compute and normalize edge weights
    weights = delta/node_dists
    edge_weight = torch.min(torch.ones_like(weights), weights)
    #edge_weight = edge_weight/edge_weight.sum(1, keepdim=True)
    edge_weight = edge_weight.view(-1)
    # Return
    return edge_index, edge_weight

def get_batch_edges(adj_matrix, threshold):
    # Initialize list of edge indices and next index (for next batch)
    edge_index_list = []
    next_idx = 0
    # Iterate over batches
    for i in range(adj_matrix.shape[0]):
        # Compute edge indices
        edge_index = (adj_matrix[i] > threshold).nonzero().t()
        # Append to list
        edge_index_list.append(edge_index + next_idx)
        # Update next index
        next_idx += adj_matrix.shape[1]
    # Concatenate edge indices
    edge_index = torch.cat(edge_index_list, 1)
    # Compute edge weights
    edge_weight = adj_matrix[adj_matrix > threshold]
    # Return
    return edge_index, edge_weight

def azim_proj_dist(xyz):
    '''
    Args:
    - xyz: tensor of size Nx3 (Cx3 or Lx3)
    - dist (string): type of distance to be returned ("l1" or "l2")
    '''
    # Convert all points to lat/lon
    x2_y2 = xyz[:,0]**2 + xyz[:,1]**2
    lat = torch.atan2(xyz[:,2], torch.sqrt(x2_y2))
    lon = torch.atan2(xyz[:,1], xyz[:,0])
    # Compute all possible pairs (first point is center)
    lat_1, lat_2 = torch.meshgrid(lat, lat)
    lon_1, lon_2 = torch.meshgrid(lon, lon)
    # Project
    cos_c = torch.sin(lat_1)*torch.sin(lat_2) + torch.cos(lat_1)*torch.cos(lat_2)*torch.cos(lon_2 - lon_1)
    c = torch.acos(cos_c)
    sin_c = torch.sin(c)
    k = c/sin_c
    k[torch.isnan(k)] = 0
    x = k*torch.cos(lat_2)*torch.sin(lon_2 - lon_1)
    y = k*(torch.cos(lat_1)*torch.sin(lat_2) - torch.sin(lat_1)*torch.cos(lat_2)*torch.cos(lon_2 - lon_1))
    # Compute L1 distances
    d = torch.abs(x) + torch.abs(y)
    # Return
    return d

def azim_proj_dist_pair(xyz_new, xyz):
    '''
    Args:
    - xyz_new: tensor of size Lx3
    - xyz: tensor of size Cx3
    '''
    # Get new number of channels
    new_num_channels = xyz_new.shape[0]
    # Concatenate points along channel dimension
    xyz = torch.cat([xyz_new, xyz], 0)
    # Convert all points to lat/lon
    x2_y2 = xyz[:,0]**2 + xyz[:,1]**2
    lat = torch.atan2(xyz[:,2], torch.sqrt(x2_y2))
    lon = torch.atan2(xyz[:,1], xyz[:,0])
    # Compute all possible pairs (first point is center)
    lat_1, lat_2 = torch.meshgrid(lat, lat)
    lon_1, lon_2 = torch.meshgrid(lon, lon)
    # Select top-right LxC matrix
    lat_1 = lat_1[:new_num_channels,new_num_channels:]
    lat_2 = lat_2[:new_num_channels,new_num_channels:]
    lon_1 = lon_1[:new_num_channels,new_num_channels:]
    lon_2 = lon_2[:new_num_channels,new_num_channels:]
    # Project
    cos_c = torch.sin(lat_1)*torch.sin(lat_2) + torch.cos(lat_1)*torch.cos(lat_2)*torch.cos(lon_2 - lon_1)
    c = torch.acos(cos_c)
    sin_c = torch.sin(c)
    k = c/sin_c
    k[torch.isnan(k)] = 0
    x = k*torch.cos(lat_2)*torch.sin(lon_2 - lon_1)
    y = k*(torch.cos(lat_1)*torch.sin(lat_2) - torch.sin(lat_1)*torch.cos(lat_2)*torch.cos(lon_2 - lon_1))
    # Compute L1 distances
    d = torch.abs(x) + torch.abs(y)
    # Return
    return d

def pairwise_euclidean_dist(coord):
    '''
    Args:
    - coord: tensor of size Nx3
    '''
    # Get number of points
    n = coord.shape[0]
    # Get numbers from 0 to n-1
    indices = torch.arange(n)
    # Compute all possible pairs
    x, y = torch.meshgrid(indices, indices)
    # Compute distances
    dist = torch.sqrt(((coord[x] - coord[y])**2).sum(2))
    # Return
    return dist
