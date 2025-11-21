import mediapipe as mp
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

N_HAND_LANDMARKS=21
N_UPPER_POSE_LANDMARKS=25
N_LANDMARKS=N_HAND_LANDMARKS*2 + N_UPPER_POSE_LANDMARKS

def mediapipe_detection(image, model):
    """Convert color space and run Mediapipe model."""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """ """
    
    # Hands extraction
    left_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(N_HAND_LANDMARKS*3)
    right_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(N_HAND_LANDMARKS*3)
    
    # Upper pose extraction
    pose_landmarks = []
    if results.pose_landmarks:
        for i, res in enumerate(results.pose_landmarks.landmark):
            if i < N_UPPER_POSE_LANDMARKS: # only capture 25 landmarks in upper body
                pose_landmarks.append([res.x, res.y, res.z])
        pose_landmarks = np.array(pose_landmarks).flatten()
    else:
        pose_landmarks = np.zeros(N_UPPER_POSE_LANDMARKS*3)
    
    return np.concatenate([left_hand_landmarks, right_hand_landmarks, pose_landmarks])

def process_video_to_sequence(path_to_video, model):
    """Processing one video to keypoints sequence through Mediapipe detection."""

    keypoints_sequence = []
    
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path_to_video}")
    
    while True:
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break
        # Make detections
        image, results = mediapipe_detection(frame, model)
        keypoints = extract_keypoints(results)
        if np.all(keypoints[:N_HAND_LANDMARKS*2] == 0.0):
            # print(f" WARNING: No hand detection.")
            continue
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
    cap.release()
    return keypoints_sequence

def interpolate_keypoints_sequence(keypoints_sequence, target_len=30):
    """Interpolate keypoints sequence into new sequence which has 60 frames for output."""
    
    sequence = np.array(keypoints_sequence.copy())
    n_frames, n_features = sequence.shape
    
    if n_frames == target_len:
        return sequence.copy()
    
    origin_indices = np.arange(n_frames)
    new_indices = np.linspace(0, n_frames - 1, target_len)
    
    interp_sequence = np.zeros((target_len, n_features))
    for i in range(n_features):
        interp_sequence[:, i] = np.interp(new_indices, origin_indices, sequence[:, i])
        
    return interp_sequence