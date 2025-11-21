import os
import json
import numpy as np
import mediapipe as mp
from src.data.process import process_video_to_sequence


def extract_keypoints_from_videos(input_video_dir="dataset/2011_videos", output_keypoint_dir="dataset/2011_keypoints", holistic=None, min_frames=25):
    """
    Extract keypoints from raw videos and save as .npy dataset.

    Args:
        input_video_dir (str): Path to original videos grouped by actions.
        output_keypoint_dir (str): Where to save extracted .npy sequences.
        holistic: Mediapipe holistic model (must be initialized before call).
        min_frames (int): Minimum frames required to save the sequence.
    """

    actions = []
    n_action = 1

    for action in os.listdir(input_video_dir):
        if not os.path.isdir(os.path.join(input_video_dir, action)):
            continue

        print(f"{n_action}. Processing action '{action}'...")
        action_dir = os.path.join(input_video_dir, action)

        file_index = 0
        for file_name in os.listdir(action_dir):
            if file_name == ".DS_Store":
                continue

            video_path = os.path.join(action_dir, file_name)
            if not os.path.isfile(video_path):
                continue

            try:
                keypoints_sequence = process_video_to_sequence(video_path, holistic)
            except Exception as e:
                print(f" ERROR: Failed to extract keypoints from {video_path}. Reason: {e}")
                continue

            # too few frames
            if keypoints_sequence is None or len(keypoints_sequence) < min_frames:
                print(f"  Skipped: Only {len(keypoints_sequence) if keypoints_sequence is not None else 0} frames.")
                continue

            # save .npy
            save_dir = os.path.join(output_keypoint_dir, action)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{file_index}.npy")
            np.save(save_path, keypoints_sequence)
            file_index += 1

        if file_index > 0:
            actions.append(action)

        print(f"  → Saved {file_index} .npy files for action '{action}'.")
        n_action += 1

    # Save label_map.json
    label_map = {label: idx for idx, label in enumerate(sorted(actions))}
    labelmap_path = os.path.join(output_keypoint_dir, "label_map.json")

    with open(labelmap_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    print("\n=== PROCESSING DONE ===")
    print(f"Collected actions: {len(actions)}")
    print(f"Label map saved to: {labelmap_path}")


if __name__ == "__main__":
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    extract_keypoints_from_videos(
        input_video_dir="dataset/2011_videos",
        output_keypoint_dir="dataset/2011_keypoints",
        holistic=holistic
    )
