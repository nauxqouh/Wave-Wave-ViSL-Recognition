import matplotlib.pyplot as plt
import numpy as np
import math

# Determine global landmarks feature
N_HAND_LANDMARKS=21
N_UPPER_POSE_LANDMARKS=25
N_LANDMARKS=N_HAND_LANDMARKS*2 + N_UPPER_POSE_LANDMARKS

# MediaPipe Hands connections (21 points)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         # thumb
    (0,5), (5,6), (6,7), (7,8),         # index 
    (5,9), (9,10), (10,11), (11,12),    # middle
    (9,13), (13,14), (14,15), (15,16),  # ring
    (13,17), (17,18), (18,19), (19,20), # pinky
    (0,17)                              # palm connection
]

# MediaPipe Pose (upper-body only)
POSE_UPPER_CONNECTIONS = [
    (11,12),   # shoulders
    (11,13), (13,15),    # left arm
    (12,14), (14,16),    # right arm
    (23,11), (24,12),    # hips to shoulders
    (23, 24) # hips
]

def draw_connections(ax, points, connections, color, linewidth=1.5):
    for i, j in connections:
        if i < len(points) and j < len(points):
            p1, p2 = points[i], points[j]
            if not np.all(p1 == 0) and not np.all(p2 == 0):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)
                
def plot_keypoints_sequence_with_connections(sequence, n_cols=5, invert_y=True):
    n_frames = len(sequence)
    n_rows = math.ceil(n_frames / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()

    def filter_zero(arr):
        arr = np.array(arr)[:, :2]
        return arr

    for idx, frame in enumerate(sequence):
        
        ax = axes[idx]
        ax.set_facecolor('#E5E5E5')
        
        left_hand_landmarks = frame[0:N_HAND_LANDMARKS*3].reshape(N_HAND_LANDMARKS, 3)
        right_hand_landmarks = frame[N_HAND_LANDMARKS*3:N_HAND_LANDMARKS*3*2].reshape(N_HAND_LANDMARKS, 3)
        pose_landmarks = frame[N_HAND_LANDMARKS*3*2:].reshape(N_UPPER_POSE_LANDMARKS, 3)

        # left_full  = filter_zero(frame[0])
        # right_full = filter_zero(frame[1])
        # body_full  = filter_zero(frame[2])
        
        left_full  = filter_zero(left_hand_landmarks)
        right_full = filter_zero(right_hand_landmarks)
        body_full  = filter_zero(pose_landmarks)

        left  = left_full[~np.all(left_full == 0, axis=1)]
        right = right_full[~np.all(right_full == 0, axis=1)]
        body  = body_full[~np.all(body_full == 0, axis=1)]

        # --- Scatter ---
        ax.scatter(left[:,0],  left[:,1],  c='red',   s=12)
        ax.scatter(right[:,0], right[:,1], c='blue',  s=12)
        ax.scatter(body[:,0],  body[:,1],  c='green', s=18)

        # --- Connections ---
        draw_connections(ax, left_full,  HAND_CONNECTIONS, 'black')
        draw_connections(ax, right_full, HAND_CONNECTIONS, 'black')
        draw_connections(ax, body_full,  POSE_UPPER_CONNECTIONS, 'black')

        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-5, 2)
        if invert_y:
            ax.invert_yaxis()

        ax.set_title(f"Frame {idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    #Ẩn subplot thừa
    for j in range(idx+1, len(axes)):
        axes[j].axis()

    plt.tight_layout()
    plt.show()