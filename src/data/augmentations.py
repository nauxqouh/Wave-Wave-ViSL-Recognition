import random
import numpy as np
import math


# Determine global landmarks feature
N_HAND_LANDMARKS=21
N_UPPER_POSE_LANDMARKS=25
N_LANDMARKS=N_HAND_LANDMARKS*2 + N_UPPER_POSE_LANDMARKS

def scale_keypoints_sequence(keypoints_sequence, scale_range=(0.9, 1.26), n_landmarks=N_LANDMARKS, n_hand_landmarks=N_HAND_LANDMARKS, n_pose_landmarks=N_UPPER_POSE_LANDMARKS):
    """
    Apply 2D scaling transformation for a sequence of keypoints.
    _random scale weight is used consistently for all frames in sequence.
    _center point to scale is computed by median based on the keypoints specified in that frame.
    """
    
    results = []
    if keypoints_sequence is None:
        return keypoints_sequence
    
    # Determine a random scale weight for all frames in sequence of keypoints
    w_scale = random.uniform(scale_range[0], scale_range[1])

    if w_scale <= 0:
        print(f"WARNING: Scale weight must be positive.")
        return keypoints_sequence
    
    for frame_keypoints in keypoints_sequence:
        if frame_keypoints is None:
            results.append(frame_keypoints.copy())
            continue
        
        if not isinstance(frame_keypoints, np.ndarray) or frame_keypoints.shape != (n_landmarks*3,):
            results.append(frame_keypoints.copy())
            continue
        
        flat_points = frame_keypoints.copy()
        try:
            points_3d = flat_points.reshape(n_landmarks,3)
        except ValueError:
            results.append(frame_keypoints.copy())
            continue   
        
        # Approximate body center
        x_center, y_center = 0.0, 0.0
        flag = False
        selected_points = points_3d[0:n_landmarks-2] # not using left-hip and right-hip
        if selected_points is not None:
            center_mask = np.any(selected_points != 0, axis=1)
            valid_points = selected_points[center_mask]
            if valid_points.shape[0] > 0:
                x_center = np.median(valid_points[:, 0])
                y_center = np.median(valid_points[:, 1])
                flag = True
            else:
                print(f"WARNING: No point is available for calculating center point.")
            
        if flag:
            # only scale points that not (0, 0, 0)
            scale_mask = np.any(points_3d != 0, axis=1)
            if np.any(scale_mask):
                x_points = points_3d[scale_mask, 0]
                y_points = points_3d[scale_mask, 1]
                
                x_trans = x_points - x_center
                y_trans = y_points - y_center
                x_scaled = x_trans * w_scale
                y_scaled = y_trans * w_scale
                x_new_points = x_scaled + x_center
                y_new_points = y_scaled + y_center
                
                points_3d[scale_mask, 0] = x_new_points
                points_3d[scale_mask, 1] = y_new_points
        # if flag=Fasle: points_3d have no changed.
    

        processed_output = points_3d.flatten()

        if np.isnan(processed_output).any() or np.isinf(processed_output).any():
            results.append(frame_keypoints.copy()) # Keep the same origin frame if NaN/Inf error
        else:
            results.append(processed_output)
        
    return results   

def rotate_keypoints_sequence(keypoints_sequence, angle_range=(-10,10), n_landmarks=N_LANDMARKS, n_hand_landmarks=N_HAND_LANDMARKS, n_pose_landmarks=N_UPPER_POSE_LANDMARKS):
    """
    Apply 2D rotation transformation for a sequence of keypoints.
    _random angle is used consistently for all frames in sequence.
    _center point to rotate is computed by median based on the keypoints specified in that frame.
    """
    results = []
    if keypoints_sequence is None:
        return keypoints_sequence
    
    # Determine a random angle for all frames in sequence of keypoints
    d_angle = random.uniform(angle_range[0], angle_range[1])
    
    # compute sin/cos with angle in radian
    r_angle = math.radians(d_angle)
    cos_angle = math.cos(r_angle)
    sin_angle = math.sin(r_angle)
    
    for frame_keypoints in keypoints_sequence:
        if frame_keypoints is None:
            results.append(frame_keypoints.copy())
            continue
        
        if not isinstance(frame_keypoints, np.ndarray) or frame_keypoints.shape != (n_landmarks*3,):
            results.append(frame_keypoints.copy())
            continue
        
        flat_points = frame_keypoints.copy()
        try:
            points_3d = flat_points.reshape(n_landmarks,3)
        except ValueError:
            results.append(frame_keypoints.copy())
            continue
        
        # Approximate body center
        x_center, y_center = 0.0, 0.0
        flag = False
        selected_points = points_3d[0:n_landmarks-2] # not using left-hip and right-hip
        if selected_points is not None:
            center_mask = np.any(selected_points != 0, axis=1)
            valid_points = selected_points[center_mask]
            if valid_points.shape[0] > 0:
                x_center = np.median(valid_points[:, 0])
                y_center = np.median(valid_points[:, 1])
                flag = True
            else:
                print(f"WARNING: No point is available for calculating center point.")
        
        if flag:
            # only rotate points that not (0, 0, 0)
            rotate_mask = np.any(points_3d != 0, axis=1)
            if np.any(rotate_mask):
                x_points = points_3d[rotate_mask, 0]
                y_points = points_3d[rotate_mask, 1]
                
                x_trans = x_points - x_center
                y_trans = y_points - y_center
                
                x_rotated = x_trans * cos_angle - y_trans * sin_angle
                y_rotated = x_trans * sin_angle + y_trans * cos_angle

                x_new_points = x_rotated + x_center
                y_new_points = y_rotated + y_center
                
                points_3d[rotate_mask, 0] = x_new_points
                points_3d[rotate_mask, 1] = y_new_points
        # if flag=Fasle: points_3d have no changed.

        processed_output = points_3d.flatten()

        if np.isnan(processed_output).any() or np.isinf(processed_output).any():
            results.append(frame_keypoints.copy()) # Keep the same origin frame if NaN/Inf error
        else:
            results.append(processed_output)
        
    return results

def translate_keypoints_sequence(keypoints_sequence, trans_x_range = (-0.05, 0.05), trans_y_range = (-0.05, 0.05),
                                 n_landmarks= N_LANDMARKS, n_hand_landmarks=N_HAND_LANDMARKS, n_pose_landmarks=N_UPPER_POSE_LANDMARKS):
    """
    Apply 2D translation transformation for a sequence of keypoints.
    _random translated weight is used consistently for all frames in sequence.
    """
    results = []
    if keypoints_sequence is None:
        return keypoints_sequence
    
    # Determine a random angle for all frames in sequence of keypoints
    dx = random.uniform(trans_x_range[0], trans_x_range[1])
    dy = random.uniform(trans_y_range[0], trans_y_range[1])

    
    for frame_keypoints in keypoints_sequence:
        if frame_keypoints is None:
            results.append(frame_keypoints.copy())
            continue
        
        if not isinstance(frame_keypoints, np.ndarray) or frame_keypoints.shape != (n_landmarks*3,):
            results.append(frame_keypoints.copy())
            continue
        
        flat_points = frame_keypoints.copy()
        try:
            points_3d = flat_points.reshape(n_landmarks,3)
        except ValueError:
            results.append(frame_keypoints.copy())
            continue
        
        # only translate points that not (0, 0, 0) # x_new = x_old + dx
        trans_mask = np.any(points_3d != 0, axis=1)
        if np.any(trans_mask):
            points_3d[trans_mask, 0] += dx 
            points_3d[trans_mask, 1] += dy
            
        
        processed_output = points_3d.flatten()

        if np.isnan(processed_output).any() or np.isinf(processed_output).any():
            results.append(frame_keypoints.copy()) # Keep the same origin frame if NaN/Inf error
        else:
            results.append(processed_output)
        
    return results

def time_stretch_keypoints_sequence(keypoints_sequence, speed_range= (0.8, 1.2)):
    """
    Change the speed of a sequence of keypoints by resampling frames.
    _a random or fixed speed_range is applied.
    """
    
    results = []
    if keypoints_sequence is None or all(frame is None for frame in keypoints_sequence):
        return keypoints_sequence
    
    valid_frames = [frame for frame in keypoints_sequence if frame is not None]
    n_origin_frames = len(valid_frames)
    
    w_speed = random.uniform(speed_range[0], speed_range[1])
    if w_speed == 1.0: # nothing to change
        return keypoints_sequence
    
    n_new_frames = int(round(n_origin_frames / w_speed))
    if n_new_frames == 0:
        if n_origin_frames > 0:
            results.append(valid_frames[0].copy() if valid_frames[0] is not None else None)
        return results
    
    # Create frame index for sampling
    original_indices = np.linspace(0, n_origin_frames - 1, n_new_frames)
    resampled_indices = np.round(original_indices).astype(int)
    resampled_indices = np.clip(resampled_indices, 0, n_origin_frames - 1)
    
    for res_idx in resampled_indices:
        results.append(valid_frames[res_idx].copy())
        
    return results