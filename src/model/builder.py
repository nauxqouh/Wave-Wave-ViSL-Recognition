import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from src.data.process import interpolate_keypoints_sequence


TRAIN_DATA = "dataset/2011_train" 
LABEL_MAP_FILE = "dataset/2011_keypoints/label_map.json" 


def load_label_map():
    if not os.path.exists(LABEL_MAP_FILE):
        raise FileNotFoundError(f"label_map.json not found at: {LABEL_MAP_FILE}")

    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return label_map

def load_dataset():
    sequences, labels = [], []
    label_map = load_label_map()

    for action in sorted(os.listdir(TRAIN_DATA)):
        action_dir = os.path.join(TRAIN_DATA, action)
        if not os.path.isdir(action_dir):
            continue

        for file_name in os.listdir(action_dir):
            if file_name == ".DS_Store":
                continue

            npy_path = os.path.join(action_dir, file_name)
            sequence = np.load(npy_path, allow_pickle=True)

            # interpolate length
            interp_seq = interpolate_keypoints_sequence(sequence)

            sequences.append(interp_seq)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)

    return X, y

def build_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(64,  return_sequences=False, dropout=0.3)))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())
    return model