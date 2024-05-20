# ECG-signal-Classification

# ECG Signal Classification using Deep Learning

## Overview

This project aims to classify ECG (Electrocardiogram) signals into different categories using deep learning techniques. The classification task involves identifying different cardiac arrhythmias based on the patterns present in the ECG signals.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [European ST-TDatabase](#European ST-TDatabase)


## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/ecg-signal-classification.git
cd ecg-signal-classification
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the necessary ECG signal data in a compatible format.
2. Run the main script to preprocess the data, train the deep learning model, and evaluate its performance.

```bash
python main.py
```

## Features

- Preprocessing of ECG signal data, including noise removal and feature extraction.
- Implementation of various deep learning models (e.g., LSTM, SimpleRNN) for classification.
- Evaluation of model performance using metrics such as accuracy, sensitivity, and specificity.
- Visualization of model results, including confusion matrices and ROC curves.

## Dependencies

- wfdb: Waveform Database library for loading and processing ECG signals.
- numpy: For numerical operations and array manipulation.
- pandas: For data manipulation and analysis.
- matplotlib: For plotting graphs and visualizations.
- tensorflow, keras: Deep learning libraries for building and training neural networks.
- scipy: For signal processing and mathematical operations.
- scikit-learn: For data preprocessing and evaluation metrics.
# European ST-T Database

This repository contains the European ST-T Database version 1.0.0, which comprises records of electrocardiogram (ECG) signals for various patients.

## File Paths

The file paths for the annotation files (.atr) within the dataset are generated using the following Python code:

```python
folder_path = 'european-st-t-database-1.0.0/european-st-t-database-1.0.0'
file_paths = gb.glob(folder_path + '/*.atr')
file_paths = [os.path.splitext(path)[0].replace("\\", "/") for path in file_paths]
# ECG Record Information

To retrieve information about a specific electrocardiogram (ECG) record, you can use the following Python code:

```python
import wfdb

# Specify the record name
record_name = 'european-st-t-database-1.0.0/european-st-t-database-1.0.0/e0103'

# Read the ECG record
record = wfdb.rdrecord(record_name)

# Print record information
print(record._dict_)

# ECG Signal Length

To retrieve the length of the electrocardiogram (ECG) signal from a specific record, you can use the following Python code:

```python
import wfdb

# Specify the record name
record_name = 'european-st-t-database-1.0.0/european-st-t-database-1.0.0/e0103'

# Read the ECG signal
ecg_signal, _ = wfdb.rdsamp(record_name)

# Print the length of the ECG signal
print(len(ecg_signal))
# ECG Annotation Information

To retrieve annotation information for a specific electrocardiogram (ECG) record, you can use the following Python code:

```python
import wfdb

# Specify the record name
record_name = 'european-st-t-database-1.0.0/european-st-t-database-1.0.0/e0103'

# Read the annotations
annotation = wfdb.rdann(record_name, 'atr')

# Print annotation information
print(annotation._dict_)
# ECG Signal Denoising using Wavelet Transform

To denoise an electrocardiogram (ECG) signal using wavelet transform, you can use the following Python code:

```python
import pywt

# Specify the desired wavelet
wavelet = 'db6'

# Decompose the ECG signal using wavelet transform
coeffs = pywt.wavedec(ecg_signal, wavelet)

# Set a threshold for noise removal
threshold = 0.5  # Adjust according to your signal characteristics

# Apply thresholding to remove noise
denoised_coeffs = [pywt.threshold(c, threshold) for c in coeffs]

# Reconstruct the denoised signal
denoised_signals = pywt.waverec(denoised_coeffs, wavelet)

# Print the denoised coefficients and signals
print("Denoised Coefficients:")
print(denoised_coeffs)
print("Denoised Signals:")
print(denoised_signals)
# Visualization of Original and Denoised ECG Signals

To visualize both the original and denoised electrocardiogram (ECG) signals, you can use the following Python code:

```python
import matplotlib.pyplot as plt

# Plot the original ECG signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_signal[:2000, 0])
plt.title('Original ECG Signal')

# Plot the denoised ECG signal
plt.subplot(2, 1, 2)
plt.plot(denoised_signals[0:2000, 0])
plt.title('Denoised ECG Signal')
# Signal Normalization

To normalize an electrocardiogram (ECG) signal, you can use the following Python code:

```python
import numpy as np

# Normalize the ECG signal
normalized_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
