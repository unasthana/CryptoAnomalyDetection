"""
Module to process and load time series data for anomaly detection.

This module includes functions for loading data, processing time series into segments,
and creating data loaders for training and testing machine learning models.

"""

import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader

PLACEHOLDER_TIMEDELTA = pd.Timedelta(minutes=0)
MIN_PUMP_SIZE = 100

FEATURE_NAMES = [
    'std_rush_order',
    'avg_rush_order',
    'std_trades',
    'std_volume',
    'avg_volume',
    'std_price',
    'avg_price',
    'avg_price_max',
    'hour_sin',
    'hour_cos',
    'minute_sin',
    'minute_cos',
    'delta_minutes',
]

def load_data(path):
        """
    Load time series data from a CSV file.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    
    return pd.read_csv(path, compression='gzip', parse_dates=['date'])


def get_pumps(data, segment_length, *, pad=True):
        """
    Extract pump segments from the input data.

    Parameters:
    - data (pd.DataFrame): Input data.
    - segment_length (int): Length of each segment.
    - pad (bool): Whether to pad segments to a specified length.

    Returns:
    - list: List of pump segments.
    """

    pumps = []
    skipped_row_count = 0

    for pump_index in np.unique(data['pump_index'].values):

        pump_i = data[data['pump_index'] == pump_index].copy()

        if len(pump_i) < MIN_PUMP_SIZE:

            print(f'Pump {pump_index} has {len(pump_i)} rows, skipping')
            skipped_row_count += len(pump_i)
            continue

        pump_i['delta_minutes'] = (pump_i['date'] - pump_i['date'].shift(1)).fillna(PLACEHOLDER_TIMEDELTA)
        pump_i['delta_minutes'] = pump_i['delta_minutes'].apply(lambda x: x.total_seconds() / 60)

        pump_i = pump_i[FEATURE_NAMES + ['gt']]
        pump_i = pump_i.values.astype(np.float32)

        if pad:
            pump_i = np.pad(pump_i, ((segment_length - 1, 0), (0, 0)), 'reflect')
        
        pumps.append(pump_i)


    print(f'Skipped {skipped_row_count} rows total')
    print(f'{len(pumps)} pumps')

    return pumps


def process_data(data, *, segment_length=60, remove_post_anomaly_data=False):
       """
    Process the input data into segments for training machine learning models.

    Parameters:
    - data (pd.DataFrame): Input data.
    - segment_length (int): Length of each segment.
    - remove_post_anomaly_data (bool): Whether to remove segments with post-anomaly data.

    Returns:
    - np.ndarray: Processed data in segment form.
    """

    print('Processing data...')
    print(f'Segment length: {segment_length}')
    print(f'Remove post anomaly data: {remove_post_anomaly_data}')
    print(f'Data shape: {data.shape}')

    pumps = get_pumps(data, segment_length)
    segments = []
    remove_cnt = 0

    for pump in pumps:
        for i, window in enumerate(np.lib.stride_tricks.sliding_window_view(pump, segment_length, axis=0)):

            segment = window.transpose()

            if remove_post_anomaly_data and segment[:-1, -1].sum() > 0:

                remove_cnt += 1
                continue

            segments.append(segment)

    
    if remove_post_anomaly_data:
        print(f'Removed {remove_cnt} rows with post-anomaly data')
    
    print(f'{len(segments)} rows of data after processing')
    
    return np.stack(segments)


def undersample_train_data(train_data, undersample_ratio):
        """
    Undersample the training data to balance the class distribution.

    Parameters:
    - train_data (np.ndarray): Training data.
    - undersample_ratio (float): Ratio of undersampling.

    Returns:
    - np.ndarray: Undersampled training data.
    """

    with_anomalies = train_data[:, :, -1].sum(axis=1) > 0
    mask = with_anomalies | (np.random.rand(train_data.shape[0]) < undersample_ratio)
    
    return train_data[mask]


def get_data(path, *,
             train_ratio=None,
             batch_size,
             undersample_ratio,
             segment_length,
             save=False,
             return_loaders=False):
    """
    Load and process data, and optionally create data loaders.

    Parameters:
    - path (str): Path to the data file.
    - train_ratio (float): Ratio of data used for training.
    - batch_size (int): Batch size for data loaders.
    - undersample_ratio (float): Ratio for undersampling.
    - segment_length (int): Length of each segment.
    - save (bool): Whether to save processed data.
    - return_loaders (bool): Whether to return data loaders.

    Returns:
    - np.ndarray or tuple: Processed data or data loaders.
    """

    assert os.path.exists(path)

    cached_data_path = f'{path}_{segment_length}.npy'

    if not os.path.exists(cached_data_path):

        data = process_data(load_data(path), segment_length=segment_length)
        
        if save:
            np.save(cached_data_path, data)
    else:
        
        print(f'Loading cached data from {cached_data_path}')
        data = np.load(cached_data_path)

    if return_loaders:

        assert train_ratio != None
        
        return create_loaders(data, train_ratio=train_ratio, batch_size=batch_size, undersample_ratio=undersample_ratio)
    
    return data


def create_loaders(data, *, train_ratio, batch_size, undersample_ratio):
       """
    Create data loaders for training and testing.

    Parameters:
    - data (np.ndarray): Processed data.
    - train_ratio (float): Ratio of data used for training.
    - batch_size (int): Batch size for data loaders.
    - undersample_ratio (float): Ratio for undersampling.

    Returns:
    - tuple: Train and test data loaders.
    """

    train_data, test_data = train_test_split(data, train_size=train_ratio, shuffle=False)

    print(f'Train data shape: {train_data.shape}')

    train_data = undersample_train_data(train_data, undersample_ratio)

    print(f'Train data shape after undersampling: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    print(f'{test_data[:, -1, -1].sum()} segments in test data ending in anomaly')

    train_data, test_data = torch.FloatTensor(train_data), torch.FloatTensor(test_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=500)

    return train_loader, test_loader


def create_loader(data, *, batch_size, undersample_ratio=1.0, shuffle=False, drop_last=True, generator=None, verbose=False):
        """
    Create a data loader from the input data.

    Parameters:
    - data (np.ndarray): Input data.
    - batch_size (int): Batch size for the data loader.
    - undersample_ratio (float): Ratio for undersampling.
    - shuffle (bool): Whether to shuffle the data.
    - drop_last (bool): Whether to drop the last batch if it is incomplete.
    - generator (torch.Generator): Generator for reproducibility.
    - verbose (bool): Whether to print verbose information.

    Returns:
    - DataLoader: PyTorch DataLoader object.
    """

    if 0 < undersample_ratio and undersample_ratio < 1:

        if verbose:
            print(f'Train data shape: {data.shape}')

        data = undersample_train_data(data, undersample_ratio)

        if verbose:
            print(f'Train data shape after undersampling: {data.shape}')

    data = torch.FloatTensor(data)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, generator=generator)



if __name__ == '__main__':

    get_data('./features_5S.csv.gz', train_ratio=0.8, batch_size=128, undersample_ratio=0.05, segment_length=60, save=True)
    get_data('./features_15S.csv.gz', train_ratio=0.8, batch_size=128, undersample_ratio=0.05, segment_length=60, save=True)
    get_data('./features_25S.csv.gz', train_ratio=0.8, batch_size=128, undersample_ratio=0.05, segment_length=60, save=True)
