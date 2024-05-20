# ECG-signal-Classification

# ECG Signal Classification using Deep Learning

## Overview

This project aims to classify ECG (Electrocardiogram) signals into different categories using deep learning techniques. The classification task involves identifying different cardiac arrhythmias based on the patterns present in the ECG signals.

## Data (i.e Very important note to download the Data)
The data is very huge to upload on github, Please download the data from the this link and extract it in the Project folder.
The link of the Data:
https://physionet.org/content/edb/1.0.0/

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
# Visualization of Original and Normalized ECG Signals

To visualize both the original and normalized electrocardiogram (ECG) signals, you can use the following Python code:

```python
import matplotlib.pyplot as plt

# Plot the original ECG signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_signal[:2000, 0])
plt.title('Original ECG Signal')

# Plot the normalized ECG signal
plt.subplot(2, 1, 2)
plt.plot(normalize_signal[:2000, 0])
plt.title('Normalized ECG Signal')
# ECG Beat Segmentation

To segment an electrocardiogram (ECG) signal into individual beats, you can use the following Python code:

```python
import wfdb

# Read the annotation for the specified record, limiting the samples to the entire signal length
annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=record.sig_len)

# Get the indices of the QRS complexes (peaks)
qrs_inds = annotation.sample

# Define the duration (in samples) for each beat segment
beat_duration = int(0.2 * record.fs)  # Assuming a 200 ms duration for each beat

# Split the ECG signal into individual beats
beats = []
for qrs_ind in qrs_inds:
    # Calculate the start and end points of each beat segment, ensuring they stay within the bounds of the signal
    start = max(0, qrs_ind - beat_duration // 2)  # Start of the beat segment
    end = min(qrs_ind + beat_duration // 2, record.sig_len)  # End of the beat segment
    beat = record.p_signal[start:end, 0]  # Extract the beat segment
    beat = beat.tolist()
    # If the length of the beat segment is less than 50 samples, pad it with the minimum value
    if len(beat) < 50:
        min_value = min(beat)
        for i in range(50 - len(beat)):
            beat.append(min_value)
    beats.append(beat)

print(beats)
# Conversion of ECG Beats to NumPy Array

To convert the segmented electrocardiogram (ECG) beats into a NumPy array, you can use the following Python code:

```python
import numpy as np

# Convert the beats list to a NumPy array
beats = np.array(beats)

# Display the NumPy array
print(beats)
# Annotation Symbols

To retrieve the symbols associated with the annotations of an electrocardiogram (ECG) record, you can use the following Python code:

```python
# Retrieve the symbols associated with the annotations
annotation_symbols = annotation.symbol
# Annotation Conversion for Model Input

To convert each annotation symbol to a corresponding number for easier processing by the model, you can use the following Python code:

```python
import numpy as np

# Identify unique annotation symbols
my_set_labels = set(annotation_symbols)

# Create a set of numbers for mapping
my_set_numbers = set(range(len(my_set_labels)))

# Create a dictionary to map symbols to numbers
my_dic = dict(zip(my_set_labels, my_set_numbers))

# Replace each symbol in annotation_symbols with its corresponding number from my_dic
converted_list = np.array([my_dic.get(item, item) for item in annotation_symbols])

print("Unique Annotation Symbols:", my_set_labels)
print("Corresponding Numbers:", my_set_numbers)
print("Symbol-to-Number Mapping:", my_dic)
print("Converted Annotation List:", converted_list)
# Train-Test Split for ECG Beats and Annotations

To split the electrocardiogram (ECG) beats and their corresponding converted annotations into training, testing, and validation sets, you can use the following Python code:

```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(beats, converted_list, test_size=0.3, random_state=42)

# Further split the test set into testing and validation sets
X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
Certainly! Below is a README file section for the model architecture, compilation, and training:

---

## Model Architecture, Compilation, and Training
### Model Architecture

The deep learning model consists of the following layers:

1. **Flatten Layer**: Flatten the 2D input array to convert it into a 1D array.
2. **Dense Layers**: Three dense layers with 64, 32, and 16 units respectively, each followed by a ReLU activation function.
3. **Output Layer**: Dense layer with 1 unit and a ReLU activation function, suitable for binary classification tasks.

### Compilation

The model is compiled using the following configurations:

- **Optimizer**: Adam optimizer is used to optimize the model parameters during training.
- **Loss Function**: Binary cross-entropy is chosen as the loss function, suitable for binary classification tasks.
- **Metrics**: Accuracy is used as the evaluation metric to monitor the model's performance during training.

### Training

The model is trained on the training dataset using the following settings:

- **Epochs**: 10 epochs are chosen to train the model.
- **Batch Size**: A batch size of 1 is used for training, meaning the model updates its parameters after processing each individual sample.

### Code Example

```python
from keras import models, layers

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(X_train.shape[1],)),  
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='relu')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=1)
```
# Model Predictions on Validation Set

To make predictions using a trained neural network model on the validation set (X_valid), you can use the following Python code:

```python
# Make predictions using the trained model on the validation set
predictions = model.predict(X_valid)

# Display the predictions
print("Predictions:", predictions)
Certainly! Below is a README file section for displaying the summary of the model:

---

## Model Summary

In this project, the summary of the deep learning model provides insights into its architecture and the number of parameters:

### Model Architecture

The deep learning model architecture is composed of several layers, including:

- Flatten Layer: Flattens the 2D input array to a 1D array.
- Dense Layers: Fully connected layers with specified number of units and activation functions.
- Output Layer: Dense layer with 1 unit and activation function, suitable for binary classification tasks.

### Number of Parameters

The summary also provides information about the total number of trainable parameters in the model, which indicates the complexity of the model.

### Code Example

```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, input_shape)        0         
_________________________________________________________________
dense (Dense)                (None, 64)                4160      
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 17        
=================================================================
Total params: 6,785
Trainable params: 6,785
Non-trainable params: 0
_________________________________________________________________
```

### Interpretation

- The "Layer (type)" column displays the type of each layer in the model.
- The "Output Shape" column shows the shape of the output of each layer.
- The "Param #" column indicates the number of trainable parameters in each layer.
- The "Total params" row displays the total number of trainable parameters in the model.

---
Sure! Below is a README file section explaining the prediction process and sensitivity calculation:

---

## Prediction and Sensitivity Calculation

In this section, the model predictions are made on the test set, and the sensitivity (true positive rate) is calculated based on the predicted labels compared to the ground truth labels.

### Prediction Process

1. **Model Prediction**: The model predicts labels for the test set using the `predict` method.
2. **Binary Conversion**: Predicted probabilities are converted to binary predictions (0 or 1) using a specified threshold.
3. **Sensitivity Calculation**: Sensitivity (true positive rate) is calculated based on the binary predictions compared to the ground truth labels.

### Sensitivity Calculation Formula

The sensitivity (true positive rate) is calculated using the following formula:

\[
\text{Sensitivity} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
\]

Where:
- True Positives (TP) are the instances correctly predicted as positive (1) by the model.
- False Negatives (FN) are the instances incorrectly predicted as negative (0) by the model, but are actually positive (1) according to the ground truth.

### Example Output

The predicted probabilities and binary predictions are printed, followed by the calculated sensitivity.

```plaintext
Predicted Probabilities: [0.3, 0.6, 0.8, ...]
Binary Predictions: [0, 1, 1, ...]
Sensitivity: 0.75
```

### Interpretation

- Sensitivity represents the ability of the model to correctly identify positive instances (e.g., presence of a condition) out of all actual positive instances.

# Model Specificity Calculation

To calculate the specificity (true negative rate) of the trained neural network model on the test set (X_test and Y_test), you can use the following Python code:

```python
# Predict labels for the test set
y_pred = model.predict(X_test)

# Convert predicted probabilities to binary predictions (0 or 1) using a threshold (4.1 in this case)
y_pred_binary = (y_pred > 4.1).astype(int)

# Calculate true negatives (TN) and false positives (FP)
TN = sum((y_pred_binary == 0) & (Y_test == 0))
FP = sum((y_pred_binary == 1) & (Y_test == 0))

# Calculate specificity (true negative rate)
specificity = TN / (TN + FP)

# Print the specificity
print("Specificity:", specificity)
# Confusion Matrix Visualization

To visualize the confusion matrix for the predicted and true labels on the test set (Y_test and y_pred_classes), you can use the following Python code:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_pred_classes and Y_test are the predicted and true labels, respectively

# Compute confusion matrix
cm = confusion_matrix(Y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
