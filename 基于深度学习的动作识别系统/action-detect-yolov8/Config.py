#coding:utf-8
import os

save_path = 'save_data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

pose_model_path = 'yolov8n-pose.pt'
detect_model_path = 'models/best.pt'
action_model_path = 'models/action_lstm.pt'

action_names = {0: '行走', 1: '跑步', 2: '坐下', 3: '站立', 4: '举手', 5: '摔倒', 6: '弯腰'}
action_names_en = {0: 'walking', 1: 'running', 2: 'sitting', 3: 'standing', 4: 'handraising', 5: 'falling', 6: 'bending'}
num_actions = 7

sequence_length = 30
num_keypoints = 17

keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

skeleton = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

skeleton_limb_colors = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (255, 102, 255),
    (255, 51, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (0, 255, 128), (0, 255, 128), (0, 255, 128), (0, 255, 128)
]

kpt_colors = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (255, 102, 255), (255, 51, 255),
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (0, 255, 128), (0, 255, 128), (0, 255, 128), (0, 255, 128), (0, 255, 128),
    (0, 255, 128), (0, 255, 128)
]

keypoint_conf_threshold = 0.5

names = action_names
CH_names = ['行走', '跑步', '坐下', '站立', '举手', '摔倒', '弯腰']
model_path = detect_model_path
