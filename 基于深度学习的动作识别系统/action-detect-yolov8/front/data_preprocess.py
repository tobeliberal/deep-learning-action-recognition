import os
import cv2
import numpy as np
import random
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

DATA_DIR = './data'
OUTPUT_PATH = './data/processed_data.npy'
SEQUENCE_LENGTH = 30

ACTION_LABELS = ['walking', 'running', 'sitting', 'standing', 'raising_hand', 'falling']
ACTION_TO_IDX = {label: idx for idx, label in enumerate(ACTION_LABELS)}


class EnhancedFeatureExtractor:
    def __init__(self, num_keypoints=17):
        self.num_keypoints = num_keypoints
    
    def extract(self, keypoints_seq):
        return self.extract_features(keypoints_seq)
    
    def _normalize_keypoints(self, kps):
        kps = kps[:, :2].copy()
        
        if len(kps) < 17:
            return kps
        
        left_hip = kps[11]
        right_hip = kps[12]
        hip_center = (left_hip + right_hip) / 2.0
        
        left_shoulder = kps[5]
        right_shoulder = kps[6]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        
        torso_len = np.linalg.norm(shoulder_center - hip_center)
        
        if torso_len < 1e-6:
            nose = kps[0]
            left_ankle = kps[15]
            right_ankle = kps[16]
            ankle_center = (left_ankle + right_ankle) / 2.0
            torso_len = np.linalg.norm(nose - ankle_center)
        
        if torso_len < 1e-6:
            torso_len = 1.0
        
        kps = (kps - hip_center) / torso_len
        
        return kps
    
    def extract_features(self, keypoints_seq):
        T = len(keypoints_seq)
        features_list = []
        
        for t in range(T):
            kps = keypoints_seq[t]
            norm_kps = self._normalize_keypoints(kps)
            features = self._extract_frame_features(norm_kps)
            features_list.append(features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        features_array = self._remove_outliers(features_array)
        
        motion_features = self._extract_motion_features(keypoints_seq)
        
        if motion_features.shape[0] > 0:
            motion_features = self._remove_outliers(motion_features)
            features_array = np.concatenate([features_array, motion_features], axis=1)
        
        return features_array
    
    def _remove_outliers(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] < 3:
            return features
        
        for col in range(features.shape[1]):
            col_data = features[:, col]
            q75, q25 = np.percentile(col_data, [75, 25])
            iqr = q75 - q25
            lower = q25 - 3.0 * iqr
            upper = q75 + 3.0 * iqr
            
            outlier_mask = (col_data < lower) | (col_data > upper)
            if np.any(outlier_mask):
                median_val = np.median(col_data[~outlier_mask]) if np.any(~outlier_mask) else np.median(col_data)
                features[outlier_mask, col] = median_val
        
        return features
    
    def _extract_frame_features(self, kps):
        features = []
        
        kps = kps[:, :2]
        
        features.extend(kps.flatten())
        
        if len(kps) >= 17:
            angles = self._compute_joint_angles(kps)
            features.extend(angles)
            
            distances = self._compute_body_distances(kps)
            features.extend(distances)
            
            ratios = self._compute_body_ratios(kps)
            features.extend(ratios)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_joint_angles(self, kps):
        angles = []
        
        if len(kps) >= 17:
            left_shoulder, right_shoulder = kps[5], kps[6]
            left_elbow, right_elbow = kps[7], kps[8]
            left_wrist, right_wrist = kps[9], kps[10]
            left_hip, right_hip = kps[11], kps[12]
            left_knee, right_knee = kps[13], kps[14]
            left_ankle, right_ankle = kps[15], kps[16]
            
            angles.append(self._angle_between_points(left_shoulder, left_elbow, left_wrist))
            angles.append(self._angle_between_points(right_shoulder, right_elbow, right_wrist))
            
            angles.append(self._angle_between_points(left_hip, left_knee, left_ankle))
            angles.append(self._angle_between_points(right_hip, right_knee, right_ankle))
            
            angles.append(self._angle_between_points(left_shoulder, left_hip, left_knee))
            angles.append(self._angle_between_points(right_shoulder, right_hip, right_knee))
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            torso_angle = self._compute_orientation_angle(shoulder_center, hip_center)
            angles.append(torso_angle)
        
        return angles
    
    def _compute_body_distances(self, kps):
        distances = []
        
        if len(kps) >= 17:
            nose = kps[0]
            left_shoulder, right_shoulder = kps[5], kps[6]
            left_hip, right_hip = kps[11], kps[12]
            left_knee, right_knee = kps[13], kps[14]
            left_ankle, right_ankle = kps[15], kps[16]
            left_wrist, right_wrist = kps[9], kps[10]
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            knee_center = (left_knee + right_knee) / 2
            ankle_center = (left_ankle + right_ankle) / 2
            
            distances.append(self._distance(nose, shoulder_center))
            distances.append(self._distance(shoulder_center, hip_center))
            distances.append(self._distance(hip_center, knee_center))
            distances.append(self._distance(knee_center, ankle_center))
            distances.append(self._distance(nose, ankle_center))
            
            distances.append(self._distance(left_shoulder, right_shoulder))
            distances.append(self._distance(left_hip, right_hip))
            
            distances.append(self._distance(left_wrist, nose))
            distances.append(self._distance(right_wrist, nose))
            
            distances.append(self._distance(left_wrist, left_hip))
            distances.append(self._distance(right_wrist, right_hip))
        
        return distances
    
    def _compute_body_ratios(self, kps):
        ratios = []
        
        if len(kps) >= 17:
            nose = kps[0]
            left_shoulder, right_shoulder = kps[5], kps[6]
            left_hip, right_hip = kps[11], kps[12]
            left_ankle, right_ankle = kps[15], kps[16]
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            ankle_center = (left_ankle + right_ankle) / 2
            
            torso_height = self._distance(shoulder_center, hip_center)
            leg_height = self._distance(hip_center, ankle_center)
            total_height = self._distance(nose, ankle_center)
            
            if total_height > 0:
                ratios.append(torso_height / total_height)
                ratios.append(leg_height / total_height)
            else:
                ratios.extend([0.5, 0.5])
            
            shoulder_width = self._distance(left_shoulder, right_shoulder)
            hip_width = self._distance(left_hip, right_hip)
            
            if shoulder_width > 0:
                ratios.append(hip_width / shoulder_width)
            else:
                ratios.append(1.0)
        
        return ratios
    
    def _extract_motion_features(self, keypoints_seq):
        T = len(keypoints_seq)
        
        if T < 2:
            return np.array([], dtype=np.float32)
        
        velocities = []
        accelerations = []
        
        for t in range(1, T):
            prev_kps = self._normalize_keypoints(keypoints_seq[t-1])
            curr_kps = self._normalize_keypoints(keypoints_seq[t])
            
            vel = curr_kps - prev_kps
            velocities.append(vel.flatten())
        
        velocities = np.array(velocities, dtype=np.float32)
        
        if len(velocities) > 1:
            for t in range(1, len(velocities)):
                acc = velocities[t] - velocities[t-1]
                accelerations.append(acc)
        
        if len(accelerations) > 0:
            accelerations = np.array(accelerations, dtype=np.float32)
            pad_vel = np.zeros((1, velocities.shape[1]), dtype=np.float32)
            pad_acc = np.zeros((2, accelerations.shape[1]), dtype=np.float32)
            velocities = np.vstack([pad_vel, velocities])
            accelerations = np.vstack([pad_acc, accelerations])
            motion_features = np.hstack([velocities, accelerations])
        else:
            pad_vel = np.zeros((1, velocities.shape[1]), dtype=np.float32)
            velocities = np.vstack([pad_vel, velocities])
            motion_features = velocities
        
        return motion_features
    
    def _distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _angle_between_points(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _compute_orientation_angle(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dx, dy))
        return abs(angle)
    
    def get_feature_dim(self):
        coord_dim = self.num_keypoints * 2
        angle_dim = 7
        distance_dim = 11
        ratio_dim = 3
        frame_dim = coord_dim + angle_dim + distance_dim + ratio_dim
        
        velocity_dim = self.num_keypoints * 2
        acceleration_dim = self.num_keypoints * 2
        motion_dim = velocity_dim + acceleration_dim
        
        return frame_dim + motion_dim


class URDataProcessor:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.model = None
        self.feature_extractor = EnhancedFeatureExtractor(num_keypoints=17)
    
    def load_model(self):
        if not HAS_YOLO:
            raise ImportError("需要安装 ultralytics: pip install ultralytics")
        self.model = YOLO('yolov8n-pose.pt')
        print("YOLOv8-Pose 模型已加载")
    
    def extract_keypoints_from_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return None
        
        results = self.model(image, verbose=False)
        
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.data
            if len(kps) > 0:
                return kps[0].cpu().numpy()[:, :2]
        
        return None
    
    def process_video(self, video_path, label):
        samples = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    无法打开视频: {video_path}")
            return samples
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  处理视频: {os.path.basename(video_path)} ({total_frames}帧, {fps:.1f}fps)")
        
        keypoints_seq = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 2 != 0:
                continue
            
            kp = self.extract_keypoints_from_image(frame)
            
            if kp is not None:
                keypoints_seq.append(kp)
            elif keypoints_seq:
                keypoints_seq.append(keypoints_seq[-1].copy())
        
        cap.release()
        
        if len(keypoints_seq) < 10:
            print(f"    跳过: 有效帧数不足 ({len(keypoints_seq)})")
            return samples
        
        samples = self._split_sequence(keypoints_seq, label)
        print(f"    生成 {len(samples)} 个样本")
        
        return samples
    
    def process_image_sequence(self, seq_dir, label):
        samples = []
        
        files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if not files:
            return samples
        
        keypoints_seq = []
        
        print(f"  处理序列: {os.path.basename(seq_dir)} ({len(files)} 帧)")
        
        for f in files:
            img_path = os.path.join(seq_dir, f)
            kp = self.extract_keypoints_from_image(img_path)
            
            if kp is not None:
                keypoints_seq.append(kp)
            elif keypoints_seq:
                keypoints_seq.append(keypoints_seq[-1].copy())
        
        if len(keypoints_seq) < 10:
            print(f"    跳过: 有效帧数不足 ({len(keypoints_seq)})")
            return samples
        
        samples = self._split_sequence(keypoints_seq, label)
        print(f"    生成 {len(samples)} 个样本")
        
        return samples
    
    def _split_sequence(self, sequence, label, window_size=SEQUENCE_LENGTH, stride=5):
        samples = []
        
        valid_sequence = []
        for kp in sequence:
            if kp is not None and kp.shape == (17, 2):
                valid_sequence.append(kp)
        
        if len(valid_sequence) < 10:
            return samples
        
        if len(valid_sequence) < window_size:
            padded = valid_sequence + [valid_sequence[-1]] * (window_size - len(valid_sequence))
            features = self.feature_extractor.extract_features(padded)
            samples.append((features, label))
        else:
            for i in range(0, len(valid_sequence) - window_size + 1, stride):
                sample = valid_sequence[i:i + window_size]
                if len(sample) == window_size:
                    features = self.feature_extractor.extract_features(sample)
                    samples.append((features, label))
        
        return samples
    
    def process_all_videos(self):
        if self.model is None:
            self.load_model()
        
        all_sequences = []
        all_labels = []
        
        video_files = sorted([f for f in os.listdir(self.data_dir) 
                             if f.endswith('.mp4') and f.startswith('fall')])
        
        print(f"\n处理跌倒视频文件 ({len(video_files)}个)...")
        fall_count = 0
        
        for video_file in video_files:
            video_path = os.path.join(self.data_dir, video_file)
            samples = self.process_video(video_path, ACTION_TO_IDX['falling'])
            for seq, lbl in samples:
                all_sequences.append(seq)
                all_labels.append(lbl)
            fall_count += len(samples)
        
        print(f"视频生成跌倒样本: {fall_count}")
        
        return all_sequences, all_labels
    
    def process_all_sequences(self):
        if self.model is None:
            self.load_model()
        
        all_sequences = []
        all_labels = []
        
        ur_fall_dir = os.path.join(self.data_dir, 'ur_fall')
        if not os.path.exists(ur_fall_dir):
            print("ur_fall数据集目录不存在")
            return all_sequences, all_labels
        
        print(f"\n处理UR Fall数据集...")
        print("处理跌倒图片序列...")
        fall_count = 0
        
        for item in sorted(os.listdir(ur_fall_dir)):
            item_path = os.path.join(ur_fall_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            if item.startswith('fall') and '-cam' in item and '-d' in item:
                samples = self.process_image_sequence(item_path, ACTION_TO_IDX['falling'])
                for seq, lbl in samples:
                    all_sequences.append(seq)
                    all_labels.append(lbl)
                fall_count += len(samples)
        
        print(f"图片序列生成跌倒样本: {fall_count}")
        
        print(f"\n处理日常活动图片序列...")
        adl_count = 0
        
        for item in sorted(os.listdir(ur_fall_dir)):
            item_path = os.path.join(ur_fall_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            if item.startswith('adl'):
                samples = self.process_image_sequence(item_path, ACTION_TO_IDX['standing'])
                for seq, lbl in samples:
                    all_sequences.append(seq)
                    all_labels.append(lbl)
                adl_count += len(samples)
        
        print(f"日常活动样本: {adl_count}")
        
        return all_sequences, all_labels
    
    def process_weizmann(self):
        if self.model is None:
            self.load_model()
        
        all_sequences = []
        all_labels = []
        
        weizmann_dir = os.path.join(self.data_dir, 'Weizmann Dataset')
        if not os.path.exists(weizmann_dir):
            print("Weizmann数据集目录不存在")
            return all_sequences, all_labels
        
        WEIZMANN_MAPPING = {
            'walk': 'walking',
            'run': 'running',
            'skip': 'running',
            'wave1': 'raising_hand',
            'wave2': 'raising_hand',
            'bend': 'sitting',
            'jump': None,
            'jack': None,
            'pjump': None,
            'side': None
        }
        
        print(f"\n处理Weizmann数据集...")
        
        for action_dir in sorted(os.listdir(weizmann_dir)):
            action_path = os.path.join(weizmann_dir, action_dir)
            
            if not os.path.isdir(action_path):
                continue
            
            target_action = WEIZMANN_MAPPING.get(action_dir)
            if target_action is None:
                continue
            
            label = ACTION_TO_IDX[target_action]
            
            video_files = [f for f in os.listdir(action_path) if f.endswith('.avi')]
            
            print(f"\n  处理动作: {action_dir} -> {target_action} ({len(video_files)}个视频)")
            
            for video_file in video_files:
                video_path = os.path.join(action_path, video_file)
                samples = self.process_video(video_path, label)
                for seq, lbl in samples:
                    all_sequences.append(seq)
                    all_labels.append(lbl)
        
        return all_sequences, all_labels
    
    def generate_mock_data(self, num_samples_per_class=200):
        sequences = []
        labels = []
        
        print(f"\n生成模拟数据...")
        
        sitting_poses = [
            np.array([
                [320, 80], [330, 70], [310, 70], [340, 80], [300, 80],
                [280, 160], [360, 160], [260, 240], [380, 240], [270, 300], [370, 300],
                [300, 320], [340, 320], [290, 380], [350, 380], [280, 400], [360, 400]
            ], dtype=np.float32),
            np.array([
                [400, 100], [410, 90], [390, 90], [420, 100], [380, 100],
                [360, 200], [440, 200], [340, 280], [460, 280], [350, 340], [450, 340],
                [380, 360], [420, 360], [370, 420], [430, 420], [360, 440], [440, 440]
            ], dtype=np.float32),
            np.array([
                [250, 120], [260, 110], [240, 110], [270, 120], [230, 120],
                [210, 220], [290, 220], [190, 300], [310, 300], [200, 360], [300, 360],
                [230, 380], [270, 380], [220, 440], [280, 440], [210, 460], [290, 460]
            ], dtype=np.float32),
        ]
        
        for base_pose in sitting_poses:
            for _ in range(num_samples_per_class // len(sitting_poses)):
                seq_len = random.randint(25, 35)
                sequence = np.zeros((seq_len, 17, 2), dtype=np.float32)
                
                for t in range(seq_len):
                    noise = np.random.randn(17, 2).astype(np.float32) * 3.0
                    sequence[t] = base_pose.copy() + noise
                    sequence[t, 7:11, 1] += random.uniform(-5, 5)
                    sequence[t, 9:11, 1] += random.uniform(-10, 10)
                
                if seq_len > SEQUENCE_LENGTH:
                    indices = np.linspace(0, seq_len - 1, SEQUENCE_LENGTH, dtype=int)
                    sequence = sequence[indices]
                elif seq_len < SEQUENCE_LENGTH:
                    padded = np.zeros((SEQUENCE_LENGTH, 17, 2), dtype=np.float32)
                    padded[:seq_len] = sequence
                    for i in range(seq_len, SEQUENCE_LENGTH):
                        padded[i] = sequence[-1]
                    sequence = padded
                
                features = self.feature_extractor.extract_features(sequence)
                sequences.append(features)
                labels.append(ACTION_TO_IDX['sitting'])
        
        standing_poses = [
            np.array([
                [400, 60], [410, 50], [390, 50], [420, 60], [380, 60],
                [370, 150], [430, 150], [360, 240], [440, 240], [350, 330], [450, 330],
                [380, 340], [420, 340], [370, 470], [430, 470], [360, 600], [440, 600]
            ], dtype=np.float32),
            np.array([
                [300, 50], [310, 40], [290, 40], [320, 50], [280, 50],
                [270, 140], [330, 140], [260, 230], [340, 230], [250, 320], [350, 320],
                [280, 330], [320, 330], [270, 460], [330, 460], [260, 590], [340, 590]
            ], dtype=np.float32),
        ]
        
        for base_pose in standing_poses:
            for _ in range(num_samples_per_class // len(standing_poses)):
                seq_len = random.randint(25, 35)
                sequence = np.zeros((seq_len, 17, 2), dtype=np.float32)
                
                for t in range(seq_len):
                    noise = np.random.randn(17, 2).astype(np.float32) * 2.0
                    sequence[t] = base_pose.copy() + noise
                
                if seq_len > SEQUENCE_LENGTH:
                    indices = np.linspace(0, seq_len - 1, SEQUENCE_LENGTH, dtype=int)
                    sequence = sequence[indices]
                elif seq_len < SEQUENCE_LENGTH:
                    padded = np.zeros((SEQUENCE_LENGTH, 17, 2), dtype=np.float32)
                    padded[:seq_len] = sequence
                    for i in range(seq_len, SEQUENCE_LENGTH):
                        padded[i] = sequence[-1]
                    sequence = padded
                
                features = self.feature_extractor.extract_features(sequence)
                sequences.append(features)
                labels.append(ACTION_TO_IDX['standing'])
        
        print(f"  sitting: {num_samples_per_class} 个样本")
        print(f"  standing: {num_samples_per_class} 个样本")
        print(f"模拟数据总数: {len(sequences)}")
        return sequences, labels
    
    def augment_data(self, sequences, labels, augment_factor=2):
        augmented_seq = list(sequences)
        augmented_labels = list(labels)
        
        for _ in range(augment_factor):
            for seq, label in zip(sequences, labels):
                aug_seq = self._augment_sequence(seq)
                augmented_seq.append(aug_seq)
                augmented_labels.append(label)
        
        print(f"数据增强后: {len(augmented_seq)} 个样本")
        return augmented_seq, augmented_labels
    
    def _augment_sequence(self, sequence):
        aug = sequence.copy()
        
        noise = np.random.randn(*aug.shape).astype(np.float32) * 0.02
        aug = aug + noise
        
        if random.random() > 0.5:
            aug[:, :34] = -aug[:, :34]
        
        scale = random.uniform(0.85, 1.15)
        aug = aug * scale
        
        if random.random() > 0.6:
            angle = random.uniform(-15, 15)
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            for t in range(aug.shape[0]):
                coords = aug[t, :34].reshape(-1, 2)
                cx, cy = coords.mean(axis=0)
                centered = coords - np.array([cx, cy])
                rotated = centered @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                aug[t, :34] = (rotated + np.array([cx, cy])).flatten()
        
        if random.random() > 0.6:
            tx = random.uniform(-0.1, 0.1)
            ty = random.uniform(-0.1, 0.1)
            for t in range(aug.shape[0]):
                coords = aug[t, :34].reshape(-1, 2)
                coords += np.array([tx, ty])
                aug[t, :34] = coords.flatten()
        
        if random.random() > 0.7:
            time_mask_len = random.randint(1, 3)
            start_idx = random.randint(0, aug.shape[0] - time_mask_len - 1)
            aug[start_idx:start_idx + time_mask_len] = aug[start_idx - 1] if start_idx > 0 else aug[start_idx + time_mask_len]
        
        if random.random() > 0.7:
            feature_mask_len = random.randint(1, 5)
            start_idx = random.randint(0, aug.shape[1] - feature_mask_len - 1)
            aug[:, start_idx:start_idx + feature_mask_len] = 0
        
        if random.random() > 0.8:
            speed_factor = random.choice([0.5, 0.75, 1.25, 1.5])
            T = aug.shape[0]
            new_T = max(10, int(T / speed_factor))
            indices = np.linspace(0, T - 1, new_T, dtype=int)
            aug_resampled = aug[indices]
            if new_T < T:
                pad = np.zeros((T - new_T, aug.shape[1]), dtype=np.float32)
                aug = np.vstack([aug_resampled, pad])
            elif new_T > T:
                aug = aug_resampled[:T]
            else:
                aug = aug_resampled
        
        return aug
    
    def process_kth(self):
        if self.model is None:
            self.load_model()
        
        all_sequences = []
        all_labels = []
        
        KTH_ACTION_MAPPING = {
            'walking': 'walking',
            'running': 'running',
            'jogging': 'running',
            'boxing': 'raising_hand',
            'handwaving': 'raising_hand',
            'handclapping': 'raising_hand'
        }
        
        kth_dir = os.path.join(self.data_dir, 'KTH_Dataset')
        if not os.path.exists(kth_dir):
            print("KTH数据集目录不存在")
            return all_sequences, all_labels
        
        print(f"\n处理KTH数据集...")
        
        for kth_action in sorted(os.listdir(kth_dir)):
            action_dir = os.path.join(kth_dir, kth_action)
            
            if not os.path.isdir(action_dir):
                continue
            
            target_action = KTH_ACTION_MAPPING.get(kth_action)
            if target_action is None:
                continue
            
            label = ACTION_TO_IDX[target_action]
            
            video_files = [f for f in os.listdir(action_dir) if f.endswith('.avi')]
            
            print(f"\n  处理动作: {kth_action} -> {target_action} ({len(video_files)}个视频)")
            
            action_count = 0
            for video_file in video_files:
                video_path = os.path.join(action_dir, video_file)
                samples = self.process_video(video_path, label)
                for seq, lbl in samples:
                    all_sequences.append(seq)
                    all_labels.append(lbl)
                action_count += len(samples)
            
            print(f"    生成样本数: {action_count}")
        
        print(f"\nKTH数据集总样本数: {len(all_sequences)}")
        return all_sequences, all_labels
    
    def process_all(self):
        video_sequences, video_labels = self.process_all_videos()
        
        image_sequences, image_labels = self.process_all_sequences()
        
        weizmann_sequences, weizmann_labels = self.process_weizmann()
        
        kth_sequences, kth_labels = self.process_kth()
        
        real_sequences = video_sequences + image_sequences + weizmann_sequences + kth_sequences
        real_labels = video_labels + image_labels + weizmann_labels + kth_labels
        
        print(f"\n真实数据统计: {len(real_sequences)} 个样本")
        
        mock_sequences, mock_labels = self.generate_mock_data(num_samples_per_class=60)
        
        all_sequences = real_sequences + mock_sequences
        all_labels = real_labels + mock_labels
        
        all_sequences, all_labels = self.augment_data(all_sequences, all_labels, augment_factor=2)
        
        label_counts = {}
        for lbl in all_labels:
            label_counts[ACTION_LABELS[lbl]] = label_counts.get(ACTION_LABELS[lbl], 0) + 1
        
        print(f"\n最终数据统计:")
        for label in ACTION_LABELS:
            count = label_counts.get(label, 0)
            print(f"  {label}: {count}")
        print(f"  总计: {len(all_sequences)}")
        
        return all_sequences, all_labels
    
    def save_data(self, sequences, labels, output_path=OUTPUT_PATH):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            'sequences': sequences,
            'labels': labels
        }
        np.save(output_path, data)
        print(f"\n数据已保存: {output_path}")


class ActionDataset(Dataset):
    def __init__(self, sequences, labels, split='train', val_ratio=0.2, random_state=42):
        self.sequences = sequences
        self.labels = labels
        
        indices = list(range(len(sequences)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=random_state,
            stratify=labels if len(set(labels)) > 1 else None
        )
        self.indices = train_idx if split == 'train' else val_idx
        
        print(f"[{split}] 加载 {len(self.indices)} 个样本")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        seq = self.sequences[real_idx]
        label = self.labels[real_idx]
        
        return torch.FloatTensor(seq), label


def create_dataloaders(data_dir=DATA_DIR, batch_size=16):
    data_path = OUTPUT_PATH
    
    if os.path.exists(data_path):
        print(f"加载已处理数据: {data_path}")
        data = np.load(data_path, allow_pickle=True).item()
        sequences = data['sequences']
        labels = data['labels']
    else:
        print("处理数据...")
        processor = URDataProcessor(data_dir)
        sequences, labels = processor.process_all()
        processor.save_data(sequences, labels)
    
    train_dataset = ActionDataset(sequences, labels, split='train')
    val_dataset = ActionDataset(sequences, labels, split='val')
    
    train_labels = [labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=6).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = np.array([class_weights[l] for l in train_labels], dtype=np.float32)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


if __name__ == '__main__':
    print("=" * 60)
    print("数据处理")
    print("=" * 60)
    
    processor = URDataProcessor()
    sequences, labels = processor.process_all()
    processor.save_data(sequences, labels)
    
    print("\n数据处理完成!")
