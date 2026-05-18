import numpy as np
import time
from typing import Optional, List, Tuple, Dict
from collections import Counter


class SequenceBuffer:
    def __init__(self, window_size: int = 30, num_keypoints: int = 17):
        self.window_size = window_size
        self.num_keypoints = num_keypoints
        self._buffer: List[np.ndarray] = []
        self._prev_kps: Optional[np.ndarray] = None
    
    def add_frame(self, keypoints: np.ndarray):
        kps = np.asarray(keypoints, dtype=np.float32)
        
        if kps.shape[0] != self.num_keypoints:
            if kps.shape[0] > self.num_keypoints:
                kps = kps[:self.num_keypoints]
            else:
                kps = np.pad(kps, ((0, self.num_keypoints - kps.shape[0]), (0, 0)))
        
        if kps.shape[1] == 2:
            kps = np.column_stack([kps, np.ones(len(kps))])
        
        kps = self._filter_low_confidence(kps)
        kps = self._filter_velocity_jitter(kps)
        kps = self._interpolate_missing(kps)
        
        self._prev_kps = kps.copy()
        self._buffer.append(kps)
        
        if len(self._buffer) > self.window_size + 10:
            self._buffer = self._buffer[-(self.window_size + 5):]
    
    def _filter_low_confidence(self, kps: np.ndarray) -> np.ndarray:
        if kps.shape[1] < 3:
            return kps
        
        conf = kps[:, 2]
        low_conf_mask = conf < 0.3
        
        if not np.any(low_conf_mask):
            return kps
        
        if self._prev_kps is not None:
            for i in range(len(kps)):
                if low_conf_mask[i]:
                    if self._prev_kps[i, 2] >= 0.3:
                        kps[i, :2] = self._prev_kps[i, :2]
                        kps[i, 2] = self._prev_kps[i, 2] * 0.7
                    elif len(self._buffer) > 0:
                        kps[i, :2] = self._buffer[-1][i, :2]
                        kps[i, 2] = self._buffer[-1][i, 2] * 0.6
        else:
            for i in range(len(kps)):
                if low_conf_mask[i]:
                    kps[i, 2] = 0.0
        
        return kps
    
    def _filter_velocity_jitter(self, kps: np.ndarray) -> np.ndarray:
        if self._prev_kps is None or len(self._buffer) < 2:
            return kps
        
        if kps.shape[1] < 3:
            return kps
        
        prev_positions = self._prev_kps[:, :2]
        curr_positions = kps[:, :2]
        prev_conf = self._prev_kps[:, 2]
        curr_conf = kps[:, 2]
        
        displacements = np.linalg.norm(curr_positions - prev_positions, axis=1)
        
        if len(self._buffer) >= 3:
            prev2_positions = self._buffer[-2][:, :2]
            avg_disp = np.linalg.norm(prev_positions - prev2_positions, axis=1)
            max_disp = np.maximum(avg_disp * 5.0, 50.0)
        else:
            max_disp = np.full(len(displacements), 80.0)
        
        for i in range(len(kps)):
            if displacements[i] > max_disp[i] and curr_conf[i] < 0.7:
                kps[i, :2] = prev_positions[i]
                kps[i, 2] = min(curr_conf[i], prev_conf[i]) * 0.8
        
        return kps
    
    def _interpolate_missing(self, kps: np.ndarray) -> np.ndarray:
        if kps.shape[1] < 3:
            return kps
        
        if len(self._buffer) < 2:
            return kps
        
        conf = kps[:, 2]
        zero_mask = conf < 0.05
        
        if not np.any(zero_mask):
            return kps
        
        prev_kps = self._buffer[-1]
        
        for i in range(len(kps)):
            if zero_mask[i] and prev_kps[i, 2] >= 0.1:
                kps[i, :2] = prev_kps[i, :2]
                kps[i, 2] = prev_kps[i, 2] * 0.5
        
        return kps
    
    def get_sequence(self) -> Optional[np.ndarray]:
        if len(self._buffer) < self.window_size:
            return None
        return np.array(self._buffer[-self.window_size:])
    
    def get_multi_scale_sequences(self) -> List[np.ndarray]:
        sequences = []
        
        if len(self._buffer) >= self.window_size:
            sequences.append(np.array(self._buffer[-self.window_size:]))
        
        if len(self._buffer) >= 20:
            short_seq = self._buffer[-20:]
            if len(short_seq) < self.window_size:
                pad_count = self.window_size - len(short_seq)
                padded = [short_seq[0]] * pad_count + short_seq
                sequences.append(np.array(padded))
        
        if len(self._buffer) >= 45:
            step = 2
            long_buf = self._buffer[-45:]
            downsampled = long_buf[::step]
            if len(downsampled) >= 20:
                sequences.append(np.array(downsampled[-self.window_size:]))
        
        return sequences if sequences else ([np.array(self._buffer[-self.window_size:])] if len(self._buffer) >= self.window_size else [])
    
    def is_ready(self) -> bool:
        return len(self._buffer) >= self.window_size
    
    def clear(self):
        self._buffer.clear()
        self._prev_kps = None
    
    def __len__(self) -> int:
        return len(self._buffer)


class PoseValidator:
    @staticmethod
    def validate_action(action: str, keypoints_seq: np.ndarray, 
                        all_probs: Dict[str, float]) -> Tuple[str, float]:
        if keypoints_seq is None or len(keypoints_seq) == 0:
            return action, all_probs.get(action, 0.0)
        
        latest_kps = keypoints_seq[-1]
        if latest_kps.shape[1] >= 3:
            avg_conf = np.mean(latest_kps[:, 2])
            if avg_conf < 0.2:
                return action, all_probs.get(action, 0.0) * 0.5
        
        if latest_kps.shape[0] < 17:
            return action, all_probs.get(action, 0.0)
        
        kps = latest_kps[:, :2]
        
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
        
        torso_vec = hip_center - shoulder_center
        torso_angle = np.degrees(np.arctan2(torso_vec[0], torso_vec[1]))
        torso_angle = abs(torso_angle)
        
        head_hip_ratio = 0.0
        hip_ankle_dist = np.linalg.norm(hip_center - ankle_center)
        head_ankle_dist = np.linalg.norm(nose - ankle_center)
        if head_ankle_dist > 1e-6:
            head_hip_ratio = hip_ankle_dist / head_ankle_dist
        
        wrist_above_shoulder = (left_wrist[1] < left_shoulder[1] or 
                                right_wrist[1] < right_shoulder[1])
        
        is_lying_down = torso_angle > 60 or head_hip_ratio < 0.3
        is_upright = torso_angle < 30 and head_hip_ratio > 0.4
        
        adjusted_probs = dict(all_probs)
        
        if action == 'falling' and not is_lying_down:
            if len(keypoints_seq) >= 5:
                recent_kps = keypoints_seq[-5:]
                was_upright = False
                for k in recent_kps:
                    tv = k[12, :2] - k[6, :2] if k.shape[0] > 12 else np.array([0, 1])
                    ta = abs(np.degrees(np.arctan2(tv[0], tv[1])))
                    if ta < 30:
                        was_upright = True
                        break
                if not was_upright:
                    adjusted_probs['falling'] *= 0.3
        
        if action == 'sitting' and is_lying_down:
            adjusted_probs['sitting'] *= 0.3
        
        if action == 'standing' and is_lying_down:
            adjusted_probs['standing'] *= 0.2
        
        if action == 'raising_hand' and not wrist_above_shoulder:
            adjusted_probs['raising_hand'] *= 0.4
        
        if action == 'walking' and is_lying_down:
            adjusted_probs['walking'] *= 0.2
        
        if action == 'running' and is_lying_down:
            adjusted_probs['running'] *= 0.2
        
        if is_lying_down:
            adjusted_probs['falling'] = adjusted_probs.get('falling', 0) * 2.0
        
        if is_upright and wrist_above_shoulder:
            adjusted_probs['raising_hand'] = adjusted_probs.get('raising_hand', 0) * 1.5
        
        total = sum(adjusted_probs.values())
        if total > 0:
            for k in adjusted_probs:
                adjusted_probs[k] /= total
        
        validated_action = max(adjusted_probs, key=adjusted_probs.get)
        validated_conf = adjusted_probs[validated_action]
        
        return validated_action, validated_conf


class ActionSmoother:
    TRANSITION_PENALTY = {
        'standing': {'falling': 0.3, 'running': 0.5},
        'sitting': {'falling': 0.3, 'running': 0.5, 'walking': 0.5},
        'walking': {'sitting': 0.5, 'raising_hand': 0.5},
        'running': {'sitting': 0.5, 'standing': 0.5, 'raising_hand': 0.5},
        'raising_hand': {'running': 0.5, 'walking': 0.5},
        'falling': {'running': 0.5, 'raising_hand': 0.5},
    }
    
    def __init__(self, vote_window: int = 20, min_hold_time: float = 2.0,
                 confidence_threshold: float = 0.5, fall_sensitivity: float = 0.35):
        self.vote_window = vote_window
        self.min_hold_time = min_hold_time
        self.confidence_threshold = confidence_threshold
        self.fall_sensitivity = fall_sensitivity
        
        self._predictions: List[Tuple[str, float, Dict[str, float]]] = []
        self._current_action: str = 'standing'
        self._current_confidence: float = 0.0
        self._action_start_time: float = time.time()
        self._last_log_action: str = ''
        self._last_log_time: float = 0.0
        self._log_interval: float = 2.0
        self._stable_count: int = 0
        self._stable_threshold: int = 5
        self._pose_validator = PoseValidator()
    
    def update(self, action: str, confidence: float, 
               all_probs: Dict[str, float] = None,
               keypoints_seq: np.ndarray = None) -> Tuple[str, float, bool]:
        now = time.time()
        
        if all_probs is None:
            all_probs = {action: confidence}
        
        if keypoints_seq is not None:
            action, confidence = self._pose_validator.validate_action(
                action, keypoints_seq, all_probs)
            all_probs[action] = confidence
        
        self._predictions.append((action, confidence, all_probs))
        if len(self._predictions) > self.vote_window:
            self._predictions.pop(0)
        
        if confidence < self.confidence_threshold and action != 'falling':
            return self._current_action, self._current_confidence, False
        
        voted_action, voted_conf = self._vote_with_transition_penalty()
        
        if voted_action == self._current_action:
            self._stable_count += 1
        else:
            self._stable_count = 0
        
        hold_duration = now - self._action_start_time
        should_update = False
        
        adaptive_hold = self.min_hold_time
        if voted_conf > 0.8:
            adaptive_hold = self.min_hold_time * 0.5
        elif voted_conf < 0.6:
            adaptive_hold = self.min_hold_time * 1.5
        
        if voted_action == 'falling' and voted_conf >= self.fall_sensitivity:
            if self._current_action != 'falling':
                should_update = True
        elif voted_action == self._current_action:
            self._current_confidence = voted_conf
        elif hold_duration >= adaptive_hold and self._stable_count >= self._stable_threshold:
            should_update = True
        
        if should_update:
            self._current_action = voted_action
            self._current_confidence = voted_conf
            self._action_start_time = now
        
        should_log = False
        if self._current_action != self._last_log_action:
            should_log = True
        elif now - self._last_log_time >= self._log_interval:
            should_log = True
        
        if should_log:
            self._last_log_action = self._current_action
            self._last_log_time = now
        
        return self._current_action, self._current_confidence, should_log
    
    def _vote_with_transition_penalty(self) -> Tuple[str, float]:
        if not self._predictions:
            return 'standing', 0.0
        
        action_scores = {}
        action_confs = {}
        
        recency_weights = np.linspace(0.3, 1.0, len(self._predictions))
        
        for idx, (action, conf, all_probs) in enumerate(self._predictions):
            recency_w = recency_weights[idx]
            
            transition_penalty = 1.0
            if self._current_action in self.TRANSITION_PENALTY:
                transition_penalty = self.TRANSITION_PENALTY[self._current_action].get(action, 1.0)
            
            consistency_bonus = 1.0
            if idx > 0 and self._predictions[idx-1][0] == action:
                consistency_bonus = 1.2
            
            weight = conf * recency_w * transition_penalty * consistency_bonus
            if action == 'falling':
                weight *= 1.5
            
            action_scores[action] = action_scores.get(action, 0) + weight
            if action not in action_confs:
                action_confs[action] = []
            action_confs[action].append(conf)
        
        best_action = max(action_scores, key=action_scores.get)
        avg_conf = np.mean(action_confs[best_action])
        
        return best_action, avg_conf
    
    def reset(self):
        self._predictions.clear()
        self._current_action = 'standing'
        self._current_confidence = 0.0
        self._action_start_time = time.time()
        self._last_log_action = ''
        self._last_log_time = 0.0
        self._stable_count = 0
    
    @property
    def current_action(self):
        return self._current_action
    
    @property
    def current_confidence(self):
        return self._current_confidence
