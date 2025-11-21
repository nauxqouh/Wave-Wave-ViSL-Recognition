import os
import numpy as np
import random
from tqdm import tqdm

# =============================
# IMPORT augmentation functions
# =============================
from src.data.augmentations import (
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence
)

def generate_new_data_v1(origin_sequence, augmentation_functions, n_samples: int, n_func_per_sample=3):
    """Generate multiple sequences by randomly combining augmentation functions."""
    
    new_samples = []
    if origin_sequence is None or augmentation_functions is None:
        return new_samples
    
    n_available_func = len(augmentation_functions)

    for _ in range(n_samples):

        curr_sequence = [frame.copy() if isinstance(frame, np.ndarray) else frame
                         for frame in origin_sequence]

        # chọn số lượng augmentation (tối đa n_func_per_sample)
        n_augs_to_apply = random.randint(1, min(n_func_per_sample, n_available_func))

        # chọn ngẫu nhiên các augmentation function (không trùng)
        selected_indices = random.sample(range(n_available_func), n_augs_to_apply)
        selected_funcs = [augmentation_functions[i] for i in selected_indices]

        # random thứ tự
        random.shuffle(selected_funcs)

        # áp dụng lần lượt
        for aug_func in selected_funcs:
            curr_sequence = aug_func(curr_sequence)
            if curr_sequence is None or all(frame is None for frame in curr_sequence):
                break

        # bỏ nếu lỗi / None
        if not curr_sequence or all(frame is None for frame in curr_sequence):
            continue

        new_samples.append(curr_sequence)
        
    return new_samples


KEYPOINT_DATA = os.path.join("dataset/2011_keypoints")
TRAIN_DATA = os.path.join("dataset/2011_train")

os.makedirs(TRAIN_DATA, exist_ok=True)

augmentations = [
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence
]

def main():
    for action in sorted(os.listdir(KEYPOINT_DATA)):

        path_origin_action = os.path.join(KEYPOINT_DATA, action)
        path_output_action = os.path.join(TRAIN_DATA, action)

        if not os.path.isdir(path_origin_action):
            continue

        os.makedirs(path_output_action, exist_ok=True)

        idx = 0
        file_list = os.listdir(path_origin_action)

        for file_name in tqdm(file_list, desc=f"Action {action}"):

            path_to_file = os.path.join(path_origin_action, file_name)
            origin_sequence = np.load(path_to_file, allow_pickle=True)

            # Generate 299 augmented + 1 original = 300 samples
            augmented_sequences = generate_new_data_v1(
                origin_sequence,
                augmentations,
                n_samples=299
            )
            augmented_sequences.append(origin_sequence)

            print(f"Number of augmented sequences: {len(augmented_sequences)}")

            # Save all samples
            for sequence in augmented_sequences:
                new_path = os.path.join(path_output_action, f"{idx}.npy")
                np.save(new_path, sequence)
                idx += 1

        print(f"==> Generated {idx} new samples for action [{action}] and saved to {path_output_action}")

if __name__ == "__main__":
    main()
