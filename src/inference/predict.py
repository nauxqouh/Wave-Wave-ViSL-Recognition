import numpy as np
import json
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from src.utils.visualize import plot_keypoints_sequence_with_connections
from src.data.process import process_video_to_sequence, interpolate_keypoints_sequence


def load_label_map(labelmap_path):
    with open(labelmap_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_action_from_sequence(keypoints_sequence, model, plot=False):
    """
    Predict the action label from pre-extracted keypoints sequence.
    """

    # Step 1 — Interpolate to target_len=30
    interp_sequence = interpolate_keypoints_sequence(keypoints_sequence)

    # Step 2 — Optional visualize
    if plot:
        plot_keypoints_sequence_with_connections(interp_sequence)

    # Step 3 — Expand dims (1, 30, n_features)
    processed_sample = np.expand_dims(interp_sequence, axis=0)

    # Step 4 — Predict
    preds = model.predict(processed_sample, verbose=0)
    label = np.argmax(preds)
    confidence = float(preds[0][label])

    return label, confidence


def predict_action_from_video(video_path, model, holistic, plot=False):
    """
    Predict directly from a raw video file.
    """

    # Extract keypoints
    keypoints_sequence = process_video_to_sequence(video_path, holistic)

    if keypoints_sequence is None or len(keypoints_sequence) == 0:
        raise ValueError("No keypoints extracted from video.")

    return predict_action_from_sequence(keypoints_sequence, model, plot)


if __name__ == "__main__":

    model_path = "models/best_model_2011.keras"
    labelmap_path = "dataset/2011_keypoints/label_map.json"
    video_path = "dataset/val_videos/máy tính laptop/may_tinh.mp4"

    # Load model + label_map
    model = load_model(model_path)
    label_map = load_label_map(labelmap_path)
    inv_label_map = {v: k for k, v in label_map.items()}

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    pred_label, conf = predict_action_from_video(
        video_path=video_path,
        model=model,
        holistic=holistic,
        plot=True
    )

    print("Prediction:", inv_label_map[pred_label])
    print("Confidence:", conf)
