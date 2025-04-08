#!/usr/bin/env python

# This script integrates ECG feature extraction for Chagas disease indicators into a ResNet1D-based classification model.

import torch
import torch.nn as nn
import numpy as np
import os
import neurokit2 as nk
from helper_code import *

################################################################################
# ResNet1D Model Definition (Completed)
################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3)  # 12 leads as input channels
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

################################################################################
# Feature Extraction Function (Enhanced with Debugging)
################################################################################

import numpy as np
import neurokit2 as nk
from scipy.stats import entropy

import numpy as np
import neurokit2 as nk
from scipy.stats import entropy

def extract_ecg_features(signal, sampling_rate=400):
    """
    Extracts core ECG features useful for detecting Chagas disease.
    Args:
        signal (np.ndarray): Shape (4096, 12) - 12-lead ECG signal.
        sampling_rate (int): Sampling rate in Hz (default 400).
    Returns:
        dict: Extracted features.
    """
    try:
        # Clean each lead
        cleaned_signal = np.array([nk.ecg_clean(signal[:, i], sampling_rate=sampling_rate)
                                   for i in range(12)]).T  # Shape: (4096, 12)

        # Use lead II (index 1) for feature extraction
        lead_II = cleaned_signal[:, 1]

        # Detect R-peaks
        _, rpeaks = nk.ecg_peaks(lead_II, sampling_rate=sampling_rate)
        rpeaks_indices = rpeaks.get('ECG_R_Peaks', [])

        # Compute RR intervals
        rr_intervals = np.diff(rpeaks_indices) / sampling_rate  # in seconds

        # 1. Heart Rate
        heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

        # 2. RR Interval Variability
        rr_variability = np.std(rr_intervals) if len(rr_intervals) > 1 else 0

        # 3. QRS Energy (using fixed window around R-peaks)
        qrs_energy_list = []
        for r in rpeaks_indices:
            start = max(0, r - int(0.05 * sampling_rate))
            end = min(len(lead_II), r + int(0.05 * sampling_rate))
            qrs_energy = np.sum(lead_II[start:end] ** 2)
            qrs_energy_list.append(qrs_energy)
        qrs_energy_avg = np.mean(qrs_energy_list) if qrs_energy_list else 0

        # 4. Signal Entropy
        hist, _ = np.histogram(lead_II, bins=100, density=True)
        signal_entropy = entropy(hist + 1e-8)  # epsilon to avoid log(0)

        # 5. Zero-Crossing Rate
        zero_crossings = np.where(np.diff(np.signbit(lead_II)))[0]
        zero_crossing_rate = len(zero_crossings) / len(lead_II)

        return {
            'Heart_Rate': heart_rate,
            'RR_Variability': rr_variability,
            'QRS_Energy': qrs_energy_avg,
            'Signal_Entropy': signal_entropy,
            'Zero_Crossing_Rate': zero_crossing_rate
        }

    except Exception as e:
        print(f"Error in ECG feature extraction: {e}")
        return {
            'Heart_Rate': 0,
            'RR_Variability': 0,
            'QRS_Energy': 0,
            'Signal_Entropy': 0,
            'Zero_Crossing_Rate': 0
        }



################################################################################
# Updated Signal Preprocessing
################################################################################

def load_preprocess_signal(record, sampling_rate=400, fixed_length=3516):
    signal, fields = load_signals(record)  # Shape: (4096, 12)
    num_samples, num_leads = signal.shape

    # Feature extraction on original signal (4096 samples)
    features = extract_ecg_features(signal, sampling_rate)

    # Pad or truncate for ResNet (3516 samples)
    if num_samples > fixed_length:
        signal_resnet = signal[:fixed_length, :]
    elif num_samples < fixed_length:
        pad_width = fixed_length - num_samples
        signal_resnet = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')
    else:
        signal_resnet = signal.copy()

    # Standardize per lead for ResNet
    for lead in range(num_leads):
        lead_signal = signal_resnet[:, lead]
        mean = np.mean(lead_signal)
        std = np.std(lead_signal)
        if std > 0:
            signal_resnet[:, lead] = (lead_signal - mean) / std
        else:
            signal_resnet[:, lead] = lead_signal - mean

    return signal_resnet, features

################################################################################
# Updated Dataset Class
################################################################################

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, records, data_folder):
        self.records = records
        self.data_folder = data_folder

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_folder, self.records[idx])
        signal, features = load_preprocess_signal(record_path)
        label = load_label(record_path)
        signal = torch.tensor(signal, dtype=torch.float32).permute(1, 0)  # (12, 3516)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label, features

################################################################################
# Updated Training Function
################################################################################

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')
    records = find_records(data_folder)
    if not records:
        raise FileNotFoundError('No data were provided.')

    dataset = ECGDataset(records, data_folder)
    labels = [load_label(os.path.join(data_folder, rec)) for rec in records]
    labels = np.array(labels, dtype=np.float32)
    neg_samples = (labels == 0).sum()
    pos_samples = (labels == 1).sum()
    pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

    batch_size = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ResNet1D(num_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if verbose:
        print('Training the model...')
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_signals, batch_labels, _ in dataloader:  # Ignore features during training
            batch_signals = batch_signals.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_signals).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_signals.size(0)
        epoch_loss = running_loss / len(dataset)
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    save_model(model_folder, model)
    if verbose:
        print('Done.')

################################################################################
# Updated Run Model Function
################################################################################

def run_model(record, model, verbose):
    net = model['model']
    device = model['device']
    signal, features = load_preprocess_signal(record)
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
    with torch.no_grad():
        output = net(signal).squeeze()
        probability_output = torch.sigmoid(output).item()
        binary_output = (probability_output > 0.5)
    if verbose:
        print(f"Features Extracted: {features}")
    # Return only binary_output and probability_output to match original expectation
    return binary_output, probability_output

################################################################################
# Remaining Functions
################################################################################

def load_model(model_folder, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1D(num_classes=1)
    model_filename = os.path.join(model_folder, 'model.pth')
    model.load_state_dict(torch.load(model_filename))
    model.to(device)
    model.eval()
    return {'model': model, 'device': device}

def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    model_filename = os.path.join(model_folder, 'model.pth')
    torch.save(model.state_dict(), model_filename)

if __name__ == '__main__':
    data_folder = 'path/to/data'
    model_folder = 'path/to/model'
    verbose = True
    train_model(data_folder, model_folder, verbose)
    model = load_model(model_folder, verbose)
    record = 'path/to/record'
    binary_output, probability_output, features = run_model(record, model, verbose)
    print(f'Binary Output: {binary_output}, Probability: {probability_output}')
    print(f'Extracted Features: {features}')