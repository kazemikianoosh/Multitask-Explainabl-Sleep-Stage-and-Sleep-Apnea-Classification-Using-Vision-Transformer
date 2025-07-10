from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import tensorflow as tf
#import wfdb
from tqdm import tqdm


def read_record(path):
    record = wfdb.rdrecord(path.decode("utf-8"))
    return record.p_signal.astype(np.float32)


def ds_base(df, shuffle, bs):
    ds = tf.data.Dataset.from_tensor_slices((df["file"], list(df["y"])))
    if shuffle:
        ds = ds.shuffle(len(df))
    ds = ds.map(
        lambda x, y: (tf.numpy_function(read_record, inp=[x], Tout=tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(lambda x, y: (tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), y))  # replace nan with zero
    ds = ds.map(lambda x, y: (tf.ensure_shape(x, [5000, 12]), y))
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def gen_datasets(df: pd.DataFrame, bs: int):
    train_ds = ds_base(df[~df["test"]], True, bs)
    val_ds = ds_base(df[df["test"]], False, bs)
    return train_ds, val_ds


def gen_df(database_root: Path, test_ratio=0.2):
    labels = [59931005, 164873001]
    labels_index = {snomed: i for i, snomed in enumerate(labels)}
    records = []
    failed = []
    for i in tqdm(list(database_root.glob("**/*.hea"))):
        file = i.with_suffix("").as_posix()
        try:
            record = wfdb.rdrecord(file)
        except Exception:
            failed.append(file)
            continue

        metadata = dict([i.split(": ") for i in record.comments])
        y = np.zeros(len(labels_index))
        if "Dx" in metadata:
            snomeds = map(int, metadata["Dx"].split(","))
            indices = [labels_index[i] for i in snomeds if i in labels_index]
            y[indices] = 1
        records.append({"file": file, "y": y})

    df = pd.DataFrame(records)
    df = pd.concat(
        [df[df["y"].apply(lambda y: np.sum(y) == 0)].sample(20000), df[df["y"].apply(lambda y: np.sum(y) != 0)]]
    )
    df["test"] = df["file"].apply(lambda file: crc32(bytes(file, "utf-8")) < test_ratio * 2**32)
    return df


# +
def create_tf_dataset(data, labels, batch_size, shuffle=True, index=False):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    indices = tf.range(start=0, limit=len(data), dtype=tf.int32)
    
#     if index:
#         # Create dataset with (data, labels, indices)
#         dataset = tf.data.Dataset.from_tensor_slices((data, labels, indices))
#     else:
#         # Create dataset with only (data, labels)
#         dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   # If we are not returning indices, remove them from the dataset
#     if index:
#         return dataset  # Return (data, labels, indices)
#     else:
        # Remove indices and return (data, labels)
    dataset = dataset.map(lambda *args: args[:2])
    return dataset

def gen_datasets_sleep(train_data, train_labels, test_data, test_labels, batch_size):
    train_ds = create_tf_dataset(train_data, train_labels, batch_size, shuffle=True)
    val_ds   = create_tf_dataset(test_data,  test_labels,  batch_size, shuffle=False )
    return train_ds, val_ds


# +
def create_tf_dataset_apnea(data, sleep_stage_labels, apnea_labels, batch_size, shuffle=True):
    # Create a dictionary for the labels
    labels = {
        'sleep_stage_output': sleep_stage_labels,
        'apnea_output': apnea_labels
    }
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
#     if index:
#         indices = tf.range(start=0, limit=len(data), dtype=tf.int32)
#         # Create dataset with (data, labels, indices)
#         dataset = tf.data.Dataset.from_tensor_slices((data, labels, indices))
#     else:
#         # Create dataset with only (data, labels)
#         dataset = tf.data.Dataset.from_tensor_slices((data, labels)) 
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    # Batch and prefetch the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    # If we are not returning indices, remove them from the dataset
#     if index:
#         return dataset  # Return (data, labels, indices)
#     else:
    # Remove indices and return (data, labels)
    dataset = dataset.map(lambda *args: args[:2])
    return dataset

def gen_datasets_sleep_apnea(train_data, train_sleep_labels, train_apnea_labels, test_data, test_sleep_labels, test_apnea_labels, batch_size):
    # Generate train dataset with both sleep stage and apnea labels
    train_ds = create_tf_dataset_apnea(train_data, train_sleep_labels, train_apnea_labels, batch_size, shuffle=True)
    # Generate validation dataset
    val_ds   = create_tf_dataset_apnea(test_data,  test_sleep_labels,  test_apnea_labels,  batch_size, shuffle=False )
    
    return train_ds, val_ds

