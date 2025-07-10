# Multimodal Sleep Stage and Sleep Apnea Classification
# Using Vision Transformer (ViT) with Multitask Learning
# Author: [Your Name]
# Date: [Date of upload to GitHub]

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from scipy.signal import resample
from keras import backend as K

# -----------------------------
# Reproducibility Setup
# -----------------------------
SEED = 0

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def min_max_normalize(data):
    data = np.asarray(data)
    normalized_data = np.zeros_like(data)
    has_nan = False
    for i in range(data.shape[1]):
        min_val = np.min(data[:, i])
        max_val = np.max(data[:, i])
        if max_val - min_val == 0:
            has_nan = True
        else:
            normalized_data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
    return normalized_data, has_nan

# Load preprocessed multimodal signal data and labels
train_data = np.load('Data_with_Apnea/train_data.npy')
train_label = np.load('Data_with_Apnea/train_label.npy')
test_data = np.load('Data_with_Apnea/test_data.npy')
test_label = np.load('Data_with_Apnea/test_label.npy')
train_apnea_label = np.load('Data_with_Apnea/train_apnea_label.npy')
test_apnea_label = np.load('Data_with_Apnea/test_apnea_label.npy')

# Resample to 2000 time points per sample
resampled_shape = (2000, 5)
train_data_resampled = np.zeros((train_data.shape[0], *resampled_shape))
test_data_resampled = np.zeros((test_data.shape[0], *resampled_shape))

for i in range(len(train_data)):
    train_data_resampled[i] = resample(train_data[i], 2000)
for i in range(len(test_data)):
    test_data_resampled[i] = resample(test_data[i], 2000)

# Normalize and filter invalid samples
train_data_normalized, train_invalid_indices = [], []
for idx, sample in enumerate(train_data_resampled):
    norm_sample, has_nan = min_max_normalize(sample)
    if has_nan:
        train_invalid_indices.append(idx)
    else:
        train_data_normalized.append(norm_sample)

test_data_normalized, test_invalid_indices = [], []
for idx, sample in enumerate(test_data_resampled):
    norm_sample, has_nan = min_max_normalize(sample)
    if has_nan:
        test_invalid_indices.append(idx)
    else:
        test_data_normalized.append(norm_sample)

train_data_normalized = np.array(train_data_normalized)
test_data_normalized = np.array(test_data_normalized)

train_label = np.delete(train_label, train_invalid_indices, axis=0)
train_apnea_label = np.delete(train_apnea_label, train_invalid_indices, axis=0)
test_label = np.delete(test_label, test_invalid_indices, axis=0)
test_apnea_label = np.delete(test_apnea_label, test_invalid_indices, axis=0)

print("Train shape:", train_data_normalized.shape, train_label.shape, train_apnea_label.shape)
print("Test shape:", test_data_normalized.shape, test_label.shape, test_apnea_label.shape)

# -----------------------------
# Dataset Preparation
# -----------------------------
from data import gen_datasets_sleep_apnea

num_classes = int(np.max(train_label) + 1)
num_apnea_class = int(np.max(train_apnea_label) + 1)

train_labels_one_hot = tf.keras.utils.to_categorical(train_label, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_label, num_classes)
train_apnea_label_one_hot = tf.keras.utils.to_categorical(train_apnea_label, num_apnea_class)
test_apnea_label_one_hot = tf.keras.utils.to_categorical(test_apnea_label, num_apnea_class)

train_ds, val_ds = gen_datasets_sleep_apnea(train_data_normalized, train_labels_one_hot, train_apnea_label_one_hot,
                                            test_data_normalized, test_labels_one_hot, test_apnea_label_one_hot,
                                            batch_size=64)

# -----------------------------
# Loss Functions
# -----------------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# -----------------------------
# Vision Transformer: Multitask Model
# -----------------------------
from vit import VisionTransformerMultiTask

vit = VisionTransformerMultiTask(
    patch_size=20,
    hidden_size=256,
    depth=6,
    num_heads=6,
    mlp_dim=128,
    num_classes=num_classes,
    apnea_classes=num_apnea_class,
    shared_dense_units=128,
    sd_survival_probability=0.7,
)

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001)

losses = {
    "sleep_stage_output": tf.keras.losses.CategoricalCrossentropy(),
    "apnea_output": tf.keras.losses.CategoricalCrossentropy(),
    # Alternative: "apnea_output": focal_loss(gamma=2., alpha=0.25)
}
loss_weights = {
    "sleep_stage_output": 1.0,
    "apnea_output": 2.0
}
metrics = {
    "sleep_stage_output": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    "apnea_output": [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
}

vit.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/vit_8000samples/", monitor="val_accuracy",
                                       save_best_only=True, save_weights_only=True)
]

# -----------------------------
# Training
# -----------------------------
epochs = 35
vit.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

# -----------------------------
# Inference & Evaluation Placeholder
# -----------------------------
# Add prediction, explanation (e.g. attention maps), and evaluation code here
# Example: vit.predict(), vit.evaluate(), or custom analysis
