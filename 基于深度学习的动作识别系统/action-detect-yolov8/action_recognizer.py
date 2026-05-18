#coding:utf-8
import torch
import torch.nn as nn
import numpy as np
import os
import Config


class ActionLSTM(nn.Module):

    def __init__(self, input_size=34, hidden_size=128, num_layers=2,
                 num_classes=6, dropout=0.5):
        super(ActionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out


class ActionRecognizer:

    def __init__(self, model_path=None, sequence_length=None):
        if model_path is None:
            model_path = Config.action_model_path
        if sequence_length is None:
            sequence_length = Config.sequence_length

        self.sequence_length = sequence_length
        self.num_keypoints = Config.num_keypoints
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ActionLSTM(
            input_size=self.num_keypoints * 2,
            hidden_size=256,
            num_layers=3,
            num_classes=Config.num_actions,
            dropout=0.4
        ).to(self.device)

        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"行为识别模型加载成功: {model_path}")
            except Exception as e:
                print(f"行为识别模型加载失败: {e}")

        self.buffers = {}
        self.last_actions = {}

    def update(self, track_id, keypoints, bbox):
        if track_id < 0:
            return self._heuristic_classify(keypoints, bbox)

        normalized_kpts = self._normalize_keypoints(keypoints, bbox)
        feature = normalized_kpts.flatten()

        if track_id not in self.buffers:
            self.buffers[track_id] = []

        self.buffers[track_id].append(feature)

        if len(self.buffers[track_id]) > self.sequence_length:
            self.buffers[track_id] = self.buffers[track_id][-self.sequence_length:]

        if self.model_loaded and len(self.buffers[track_id]) >= self.sequence_length:
            result = self._predict(track_id)
            if result:
                self.last_actions[track_id] = result
                return result

        if track_id in self.last_actions:
            return self.last_actions[track_id]

        return self._heuristic_classify(keypoints, bbox)

    def _predict(self, track_id):
        if track_id not in self.buffers:
            return None

        sequence = self.buffers[track_id][-self.sequence_length:]
        input_tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return {
            'action': Config.action_names[pred_class],
            'action_id': pred_class,
            'confidence': confidence
        }

    def _normalize_keypoints(self, keypoints, bbox):
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)

        normalized = keypoints.copy().astype(float)
        normalized[:, 0] = (keypoints[:, 0] - x1) / w
        normalized[:, 1] = (keypoints[:, 1] - y1) / h
        normalized = np.clip(normalized, 0, 1)

        return normalized

    def _heuristic_classify(self, keypoints, bbox):
        try:
            left_shoulder = keypoints[5].astype(float)
            right_shoulder = keypoints[6].astype(float)
            left_hip = keypoints[11].astype(float)
            right_hip = keypoints[12].astype(float)
            left_ankle = keypoints[15].astype(float)
            right_ankle = keypoints[16].astype(float)
            left_wrist = keypoints[9].astype(float)
            right_wrist = keypoints[10].astype(float)
            nose = keypoints[0].astype(float)

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            ankle_center = (left_ankle + right_ankle) / 2

            body_vec = hip_center - shoulder_center
            body_angle = abs(np.arctan2(body_vec[0], max(abs(body_vec[1]), 1)))

            x1, y1, x2, y2 = bbox
            h = max(y2 - y1, 1)
            w = max(x2 - x1, 1)
            hip_height_ratio = (y2 - hip_center[1]) / h

            body_width = np.linalg.norm(left_shoulder - right_shoulder)
            body_height = np.linalg.norm(shoulder_center - ankle_center)
            aspect_ratio = body_width / max(body_height, 1)

            wrist_above_shoulder = (left_wrist[1] < left_shoulder[1] or
                                    right_wrist[1] < right_shoulder[1])

            nose_above_hip = nose[1] < hip_center[1]
            shoulder_near_hip = abs(shoulder_center[1] - hip_center[1]) < h * 0.15

            if hip_height_ratio < 0.3 and body_angle > 0.5:
                action_id = 5
            elif hip_height_ratio < 0.35 and body_angle > 0.3:
                action_id = 5
            elif not nose_above_hip and shoulder_near_hip and body_angle > 0.4:
                action_id = 6
            elif body_angle > 0.6 and hip_height_ratio > 0.4:
                action_id = 6
            elif wrist_above_shoulder:
                action_id = 4
            elif hip_height_ratio < 0.5 and body_angle < 0.3:
                action_id = 2
            elif body_angle < 0.2 and aspect_ratio < 0.6:
                action_id = 3
            elif body_angle < 0.3:
                action_id = 0
            else:
                action_id = 1

            return {
                'action': Config.action_names[action_id],
                'action_id': action_id,
                'confidence': 0.5
            }
        except Exception:
            return {
                'action': '未知',
                'action_id': -1,
                'confidence': 0.0
            }

    def remove_stale_tracks(self, active_track_ids):
        stale_ids = [tid for tid in self.buffers if tid not in active_track_ids]
        for tid in stale_ids:
            del self.buffers[tid]
            if tid in self.last_actions:
                del self.last_actions[tid]

    def get_current_action(self, track_id):
        return self.last_actions.get(track_id, None)

    def reset(self):
        self.buffers = {}
        self.last_actions = {}
