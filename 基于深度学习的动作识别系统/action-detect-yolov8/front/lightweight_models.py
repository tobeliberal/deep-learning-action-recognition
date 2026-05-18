import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque

ACTION_LABELS = ['walking', 'running', 'sitting', 'standing', 'raising_hand', 'falling']


class LightweightGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        
        gru_out, _ = self.gru(x)
        
        attn_weights = self.attention(gru_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        return self.classifier(context)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        return self.classifier(last_out)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels)) if bias else None
        
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, T, V = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
            out = torch.matmul(x, self.weight)
            out = torch.matmul(adj, out)
            out = out.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous()
        else:
            out = torch.matmul(x, self.weight)
            out = torch.matmul(adj, out)
        
        return out + self.bias if self.bias is not None else out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels)
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, adj)
        x = self.tcn(x) + res
        return self.dropout(self.relu(x))


class LightweightSTGCN(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        hidden_channels: list = [64, 64, 128],
        num_classes: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.register_buffer('adj', self._build_adjacency(num_joints))
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        self.st_gcn_blocks = nn.ModuleList()
        channels = [in_channels] + hidden_channels
        
        for i in range(len(channels) - 1):
            self.st_gcn_blocks.append(
                STGCNBlock(channels[i], channels[i + 1], stride=1 if i < len(channels) - 2 else 2, dropout=dropout)
            )
        
        self.classifier = nn.Linear(hidden_channels[-1], num_classes)
    
    def _build_adjacency(self, num_joints: int) -> torch.Tensor:
        adj = torch.zeros(num_joints, num_joints)
        
        if num_joints == 17:
            skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        else:
            skeleton = [(i, i + 1) for i in range(num_joints - 1)]
        
        for i, j in skeleton:
            adj[i, j] = adj[j, i] = 1
        
        adj = adj + torch.eye(num_joints)
        return adj / adj.sum(dim=1, keepdim=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] != self.num_joints:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        B, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
        x = self.data_bn(x).view(B, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        for block in self.st_gcn_blocks:
            x = block(x, self.adj)
        
        x = F.avg_pool2d(x, (x.shape[2], 1)).view(B, -1)
        return self.classifier(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RuleBasedActionClassifier:
    def __init__(self, num_joints: int = 17, smooth_window: int = 5):
        self.num_joints = num_joints
        self.labels = ACTION_LABELS
        self.smooth_window = smooth_window
        self._action_history: deque = deque(maxlen=smooth_window)
        self._prev_keypoints: Optional[np.ndarray] = None
        self._velocity_history: deque = deque(maxlen=10)
        
        self._action_names = {
            'walking': '行走',
            'running': '跑步', 
            'sitting': '坐下',
            'standing': '站立',
            'raising_hand': '举手',
            'falling': '跌倒'
        }
    
    def predict(self, sequence: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if sequence.ndim == 3:
            keypoints_seq = sequence
        else:
            keypoints_seq = sequence[np.newaxis, :, :]
        
        features = self._extract_features(keypoints_seq)
        
        scores = self._classify(features)
        
        raw_action = max(scores, key=scores.get)
        
        smoothed_action = self._smooth_action(raw_action)
        
        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}
        
        confidence = probs[smoothed_action]
        
        return smoothed_action, confidence, probs
    
    def _extract_features(self, sequence: np.ndarray) -> Dict:
        features = {}
        
        last_frame = sequence[-1]
        kps = last_frame[:, :2] if last_frame.ndim == 2 else last_frame.reshape(-1, 2)
        
        if self.num_joints == 17:
            features.update(self._extract_yolo_features(kps))
        else:
            features.update(self._extract_mediapipe_features(kps))
        
        if len(sequence) > 1:
            features.update(self._extract_motion_features(sequence))
        
        return features
    
    def _extract_yolo_features(self, kps: np.ndarray) -> Dict:
        features = {}
        
        if len(kps) < 13:
            return {'valid': False}
        
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
        
        features['torso_height'] = self._distance(shoulder_center, hip_center)
        features['leg_height'] = self._distance(hip_center, ankle_center)
        features['total_height'] = self._distance(nose, ankle_center)
        features['thigh_height'] = self._distance(hip_center, knee_center)
        
        features['torso_angle'] = self._get_angle(shoulder_center, hip_center)
        features['left_leg_angle'] = self._get_angle(left_hip, left_knee)
        features['right_leg_angle'] = self._get_angle(right_hip, right_knee)
        
        features['shoulder_width'] = self._distance(left_shoulder, right_shoulder)
        features['hip_width'] = self._distance(left_hip, right_hip)
        
        features['left_hand_above_head'] = left_wrist[1] < nose[1] - 30
        features['right_hand_above_head'] = right_wrist[1] < nose[1] - 30
        features['any_hand_raised'] = features['left_hand_above_head'] or features['right_hand_above_head']
        
        features['hip_y'] = hip_center[1]
        features['knee_y'] = knee_center[1]
        features['ankle_y'] = ankle_center[1]
        
        features['valid'] = True
        
        return features
    
    def _extract_mediapipe_features(self, kps: np.ndarray) -> Dict:
        features = {}
        
        if len(kps) < 28:
            return {'valid': False}
        
        nose = kps[0]
        left_shoulder, right_shoulder = kps[11], kps[12]
        left_hip, right_hip = kps[23], kps[24]
        left_knee, right_knee = kps[25], kps[26]
        left_ankle, right_ankle = kps[27], kps[28]
        left_wrist, right_wrist = kps[15], kps[16]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        knee_center = (left_knee + right_knee) / 2
        ankle_center = (left_ankle + right_ankle) / 2
        
        features['torso_height'] = self._distance(shoulder_center, hip_center)
        features['leg_height'] = self._distance(hip_center, ankle_center)
        features['total_height'] = self._distance(nose, ankle_center)
        
        features['torso_angle'] = self._get_angle(shoulder_center, hip_center)
        features['left_leg_angle'] = self._get_angle(left_hip, left_knee)
        features['right_leg_angle'] = self._get_angle(right_hip, right_knee)
        
        features['left_hand_above_head'] = left_wrist[1] < nose[1] - 30
        features['right_hand_above_head'] = right_wrist[1] < nose[1] - 30
        features['any_hand_raised'] = features['left_hand_above_head'] or features['right_hand_above_head']
        
        features['hip_y'] = hip_center[1]
        features['knee_y'] = knee_center[1]
        
        features['valid'] = True
        
        return features
    
    def _extract_motion_features(self, sequence: np.ndarray) -> Dict:
        features = {}
        
        velocities = []
        for i in range(1, len(sequence)):
            prev = sequence[i-1, :, :2]
            curr = sequence[i, :, :2]
            vel = np.mean(np.abs(curr - prev))
            velocities.append(vel)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            max_velocity = np.max(velocities)
            features['avg_velocity'] = avg_velocity
            features['max_velocity'] = max_velocity
        else:
            features['avg_velocity'] = 0
            features['max_velocity'] = 0
        
        self._velocity_history.append(features['avg_velocity'])
        if len(self._velocity_history) >= 5:
            features['velocity_variance'] = np.var(list(self._velocity_history))
        else:
            features['velocity_variance'] = 0
        
        return features
    
    def _classify(self, features: Dict) -> Dict[str, float]:
        scores = {label: 0.0 for label in self.labels}
        
        if not features.get('valid', False):
            scores['standing'] = 0.5
            return scores
        
        torso_angle = features.get('torso_angle', 0)
        torso_height = features.get('torso_height', 0)
        total_height = features.get('total_height', 1)
        leg_height = features.get('leg_height', 1)
        hip_y = features.get('hip_y', 0)
        knee_y = features.get('knee_y', 0)
        avg_velocity = features.get('avg_velocity', 0)
        velocity_variance = features.get('velocity_variance', 0)
        
        if torso_angle > 55:
            scores['falling'] = 0.8
            scores['sitting'] = 0.1
            return scores
        
        if features.get('any_hand_raised', False):
            scores['raising_hand'] = 0.7
            scores['standing'] = 0.2
            scores['sitting'] = 0.05
            return scores
        
        sitting_ratio = torso_height / max(total_height, 1)
        hip_knee_ratio = abs(hip_y - knee_y) / max(leg_height, 1)
        
        if sitting_ratio < 0.25 or hip_knee_ratio < 0.3:
            scores['sitting'] = 0.7
            scores['standing'] = 0.15
            scores['walking'] = 0.1
        elif avg_velocity > 8:
            scores['running'] = 0.6
            scores['walking'] = 0.25
            scores['standing'] = 0.1
        elif avg_velocity > 3:
            scores['walking'] = 0.6
            scores['standing'] = 0.2
            scores['running'] = 0.1
        elif avg_velocity > 1:
            scores['walking'] = 0.4
            scores['standing'] = 0.4
            scores['sitting'] = 0.1
        else:
            scores['standing'] = 0.6
            scores['sitting'] = 0.25
            scores['walking'] = 0.1
        
        return scores
    
    def _smooth_action(self, action: str) -> str:
        self._action_history.append(action)
        
        if len(self._action_history) < 3:
            return action
        
        action_counts = {}
        for a in self._action_history:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        most_common = max(action_counts, key=action_counts.get)
        count = action_counts[most_common]
        
        if count >= len(self._action_history) * 0.4:
            return most_common
        else:
            return action
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        return min(angle, 180 - angle)
    
    def reset(self):
        self._action_history.clear()
        self._velocity_history.clear()
        self._prev_keypoints = None
    
    def count_parameters(self) -> int:
        return 0


class EnsembleActionRecognizer:
    def __init__(self, input_dim: int = 123, num_classes: int = 6, device: str = 'cpu',
                 model_dir: str = './checkpoints'):
        self.num_classes = num_classes
        self.device = device
        self.labels = ACTION_LABELS[:num_classes]
        self.models = []
        self.model_weights = []
        
        gru_model = LightweightGRU(input_dim=input_dim, num_classes=num_classes)
        gru_path = os.path.join(model_dir, 'best_model_all_datasets.pth')
        if os.path.exists(gru_path):
            gru_model.load_state_dict(torch.load(gru_path, map_location=device))
            gru_model.to(device)
            gru_model.eval()
            self.models.append(('gru', gru_model))
            self.model_weights.append(0.55)
            print(f"集成模型 - GRU已加载: {gru_path}")
        
        lstm_model = LightweightLSTM(input_dim=input_dim, num_classes=num_classes)
        lstm_path = os.path.join(model_dir, 'best_model_lstm.pth')
        if os.path.exists(lstm_path):
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            lstm_model.to(device)
            lstm_model.eval()
            self.models.append(('lstm', lstm_model))
            self.model_weights.append(0.45)
            print(f"集成模型 - LSTM已加载: {lstm_path}")
        
        if not self.models and os.path.exists(gru_path):
            gru_model2 = LightweightGRU(input_dim=input_dim, num_classes=num_classes)
            gru_model2.load_state_dict(torch.load(gru_path, map_location=device))
            gru_model2.to(device)
            gru_model2.eval()
            self.models.append(('gru2', gru_model2))
            self.model_weights.append(1.0)
            print("集成模型 - 使用GRU单模型（LSTM未找到）")
        
        total_w = sum(self.model_weights)
        self.model_weights = [w / total_w for w in self.model_weights]
        
        self.temperature = 0.5
        
        print(f"集成模型数量: {len(self.models)}, 权重: {self.model_weights}")
    
    def predict(self, sequence: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if not self.models:
            return 'standing', 0.0, {}
        
        all_probs = []
        
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for name, model in self.models:
                logits = model(x)
                scaled_logits = logits / self.temperature
                probs = F.softmax(scaled_logits, dim=1)
                all_probs.append(probs)
        
        weighted_probs = torch.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, self.model_weights):
            weighted_probs += probs * weight
        
        conf, pred = torch.max(weighted_probs, 1)
        
        return self.labels[pred.item()], conf.item(), {self.labels[i]: weighted_probs[0, i].item() for i in range(self.num_classes)}
    
    def predict_with_tta(self, sequences: list) -> Tuple[str, float, Dict[str, float]]:
        if not self.models or not sequences:
            return 'standing', 0.0, {}
        
        all_seq_probs = []
        
        for seq in sequences:
            x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            seq_probs = torch.zeros(1, self.num_classes, device=self.device)
            
            with torch.no_grad():
                for name, model in self.models:
                    logits = model(x)
                    scaled_logits = logits / self.temperature
                    probs = F.softmax(scaled_logits, dim=1)
                    idx = self.models.index((name, model))
                    seq_probs += probs * self.model_weights[idx]
            
            all_seq_probs.append(seq_probs)
        
        avg_probs = torch.stack(all_seq_probs).mean(dim=0)
        conf, pred = torch.max(avg_probs, 1)
        
        return self.labels[pred.item()], conf.item(), {self.labels[i]: avg_probs[0, i].item() for i in range(self.num_classes)}


class ActionRecognizer:
    def __init__(
        self,
        model_type: str = 'gru',
        input_dim: int = 48,
        num_joints: int = 17,
        num_classes: int = 6,
        device: str = 'cpu',
        model_path: Optional[str] = None,
        use_rule_based: bool = True
    ):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.labels = ACTION_LABELS[:num_classes]
        self.use_rule_based = use_rule_based
        
        if model_path and os.path.exists(model_path):
            use_rule_based = False
            self.use_rule_based = False
        
        if self.use_rule_based:
            self.model = RuleBasedActionClassifier(num_joints=num_joints)
            print("使用基于规则的行为分类器（演示模式）")
        else:
            if model_type == 'gru':
                self.model = LightweightGRU(input_dim=input_dim, num_classes=num_classes)
            elif model_type == 'lstm':
                self.model = LightweightLSTM(input_dim=input_dim, num_classes=num_classes)
            elif model_type == 'stgcn':
                self.model = LightweightSTGCN(num_joints=num_joints, num_classes=num_classes)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.model.to(device)
            
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"已加载训练模型: {model_path}")
            
            self.model.eval()
            print(f"模型已初始化: {model_type.upper()}")
            print(f"参数量: {self.model.count_parameters():,} ({self.model.count_parameters()/1e6:.2f}M)")
    
    def predict(self, sequence: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if self.use_rule_based:
            return self.model.predict(sequence)
        
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        if x.dim() == 3 and self.model_type == 'stgcn':
            x = x.permute(0, 2, 1).unsqueeze(1)
        
        with torch.no_grad():
            logits = self.model(x)
            temperature = 0.5
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=1)
            conf, pred = torch.max(probs, 1)
            
            return self.labels[pred.item()], conf.item(), {self.labels[i]: probs[0, i].item() for i in range(self.num_classes)}
    
    def reset(self):
        if self.use_rule_based:
            self.model.reset()
