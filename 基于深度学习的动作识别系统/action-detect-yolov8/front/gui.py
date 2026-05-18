import sys
import time
from datetime import datetime
from collections import deque
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QListWidget, QListWidgetItem, QMessageBox, QFrame, QGridLayout, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from utils.one_euro_filter import OneEuroFilter
from utils.feature_extractor import SequenceBuffer, ActionSmoother
from data_preprocess import EnhancedFeatureExtractor
from models.lightweight_models import ActionRecognizer, EnsembleActionRecognizer


STYLE_SHEET = """
QMainWindow { background-color: #0d1117; }
QWidget { font-family: 'Microsoft YaHei', sans-serif; color: #e6edf3; }
QGroupBox { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-top: 12px; padding: 15px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; color: #58a6ff; }
QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 10px 20px; font-size: 13px; color: #e6edf3; }
QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
QPushButton:disabled { background-color: #161b22; color: #484f58; }
QPushButton#startBtn { background-color: #238636; border-color: #238636; }
QPushButton#startBtn:hover { background-color: #2ea043; }
QPushButton#stopBtn { background-color: #da3633; border-color: #da3633; }
QPushButton#stopBtn:hover { background-color: #f85149; }
QComboBox, QSpinBox { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; color: #e6edf3; }
QComboBox:hover, QSpinBox:hover { border-color: #58a6ff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #8b949e; margin-right: 10px; }
QComboBox QAbstractItemView { background-color: #161b22; border: 1px solid #30363d; selection-background-color: #1f6feb; }
QCheckBox { font-size: 13px; spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 1px solid #30363d; background-color: #0d1117; }
QCheckBox::indicator:checked { background-color: #238636; border-color: #238636; }
QListWidget { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; font-size: 13px; }
QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #21262d; }
QListWidget::item:selected { background-color: #1f6feb; }
QStatusBar { background-color: #161b22; border-top: 1px solid #30363d; font-size: 12px; }
"""


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, str, float, float, float)
    action_detected = pyqtSignal(str, float, str)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(dict)
    initialized = pyqtSignal(bool)
    
    def __init__(self, source, pose_backend='mediapipe', model_type='gru', window_size=30, enable_smoothing=True):
        super().__init__()
        self.source = source
        self.pose_backend = pose_backend
        self.model_type = model_type
        self.window_size = window_size
        self.enable_smoothing = enable_smoothing
        
        self._running = False
        self._paused = False
        self._cap = None
        self._pose_estimator = None
        self._skeleton_connections = []
        self._frame_count = 0
        self._action_count = {'walking': 0, 'running': 0, 'sitting': 0, 'standing': 0, 'raising_hand': 0, 'falling': 0}
    
    def initialize(self):
        try:
            print(f"正在打开视频源: {self.source}")
            
            if isinstance(self.source, int):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self.source)
            
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self._cap.isOpened():
                print("DSHOW 后端失败，尝试默认后端...")
                if isinstance(self.source, int):
                    self._cap = cv2.VideoCapture(self.source)
                if not self._cap.isOpened():
                    self.error_occurred.emit(f"无法打开视频源 (索引: {self.source})")
                    return False
            
            print("视频源打开成功")
            
            if self.pose_backend == 'mediapipe':
                import mediapipe as mp
                mp_version = mp.__version__
                major, minor = map(int, mp_version.split('.')[:2])
                if major == 0 and minor >= 10:
                    print(f"检测到 MediaPipe {mp_version}，自动切换到 YOLOv8-Pose...")
                    self.pose_backend = 'yolov8'
            
            if self.pose_backend == 'yolov8':
                from ultralytics import YOLO
                print("正在加载 YOLOv8-Pose...")
                self._pose_estimator = YOLO('yolov8n-pose.pt')
                self.num_keypoints = 17
                self._skeleton_connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                ]
            else:
                print("正在加载 MediaPipe Pose...")
                self._mp_pose = mp.solutions.pose
                self._pose_estimator = self._mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self.num_keypoints = 33
                self._skeleton_connections = [
                    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
                ]
            
            print(f"姿态估计器已加载: {self.pose_backend}")
            
            self._smoother = OneEuroFilter(num_keypoints=self.num_keypoints) if self.enable_smoothing else None
            self._buffer = SequenceBuffer(window_size=self.window_size, num_keypoints=self.num_keypoints)
            self._feature_extractor = EnhancedFeatureExtractor(num_keypoints=self.num_keypoints)
            self._action_smoother = ActionSmoother(vote_window=20, min_hold_time=2.0, confidence_threshold=0.5, fall_sensitivity=0.35)
            
            try:
                self._recognizer = EnsembleActionRecognizer(
                    input_dim=self._feature_extractor.get_feature_dim(),
                    num_classes=6,
                    model_dir='./checkpoints'
                )
                print("使用集成模型（GRU+LSTM）")
            except Exception as e:
                print(f"集成模型加载失败: {e}，使用单模型")
                self._recognizer = ActionRecognizer(
                    model_type=self.model_type,
                    input_dim=self._feature_extractor.get_feature_dim(),
                    num_joints=self.num_keypoints
                )
            
            print("初始化完成")
            return True
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"初始化失败: {str(e)}\n{traceback.format_exc()}")
            return False
    
    def run(self):
        if not self.initialize():
            self.initialized.emit(False)
            return
        
        self.initialized.emit(True)
        self._running = True
        self._frame_count = 0
        fps_history = deque(maxlen=30)
        
        print("开始处理循环")
        
        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue
            
            start_time = time.time()
            
            try:
                ret, frame = self._cap.read()
                if not ret:
                    if isinstance(self.source, str) and not str(self.source).isdigit():
                        break
                    time.sleep(0.001)
                    continue
                
                self._frame_count += 1
                keypoints = self._detect_pose(frame)
                
                action = 'no_pose'
                confidence = 0.0
                
                if keypoints is not None:
                    if self._smoother:
                        keypoints = self._smoother.update(keypoints)
                    
                    self._buffer.add_frame(keypoints)
                    
                    if self._buffer.is_ready():
                        sequences = self._buffer.get_multi_scale_sequences()
                        
                        all_probs_list = []
                        for seq in sequences:
                            features = self._feature_extractor.extract(seq)
                            _, _, probs = self._recognizer.predict(features)
                            all_probs_list.append(probs)
                        
                        merged_probs = {}
                        for probs in all_probs_list:
                            for k, v in probs.items():
                                merged_probs[k] = merged_probs.get(k, 0) + v
                        for k in merged_probs:
                            merged_probs[k] /= len(all_probs_list)
                        
                        raw_action = max(merged_probs, key=merged_probs.get)
                        raw_conf = merged_probs[raw_action]
                        
                        current_sequence = self._buffer.get_sequence()
                        
                        smoothed_action, smoothed_conf, should_log = self._action_smoother.update(
                            raw_action, raw_conf, merged_probs, current_sequence)
                        
                        action = smoothed_action
                        confidence = smoothed_conf
                        
                        if should_log and action in self._action_count:
                            self._action_count[action] += 1
                            self.action_detected.emit(action, confidence, datetime.now().strftime("%H:%M:%S"))
                    else:
                        action = 'collecting'
                        confidence = len(self._buffer) / self.window_size
                    
                    frame = self._draw_skeleton(frame, keypoints)
                
                process_time = time.time() - start_time
                fps_history.append(1.0 / max(process_time, 0.001))
                fps = sum(fps_history) / len(fps_history)
                
                self.frame_ready.emit(frame, action, confidence, fps, process_time)
                self.status_update.emit({'frame_count': self._frame_count, 'action_count': self._action_count.copy(), 'buffer_fill': len(self._buffer) / self.window_size * 100})
                
            except Exception as e:
                print(f"处理错误: {e}")
                time.sleep(0.01)
        
        self._cleanup()
    
    def _detect_pose(self, frame):
        try:
            if self.pose_backend == 'mediapipe':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._pose_estimator.process(rgb)
                if results.pose_landmarks:
                    h, w = frame.shape[:2]
                    return np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            else:
                results = self._pose_estimator(frame, verbose=False)
                if results and results[0].keypoints is not None:
                    kps = results[0].keypoints.data
                    if len(kps) > 0:
                        return kps[0].cpu().numpy()
        except Exception as e:
            print(f"姿态检测错误: {e}")
        return None
    
    def _draw_skeleton(self, frame, keypoints):
        output = frame.copy()
        try:
            for i, j in self._skeleton_connections:
                if i < len(keypoints) and j < len(keypoints) and keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                    cv2.line(output, (int(keypoints[i, 0]), int(keypoints[i, 1])), (int(keypoints[j, 0]), int(keypoints[j, 1])), (0, 255, 136), 2)
            for kp in keypoints:
                if kp[2] > 0.3:
                    cv2.circle(output, (int(kp[0]), int(kp[1])), 4, (255, 200, 0), -1)
        except Exception as e:
            print(f"绘制骨架错误: {e}")
        return output
    
    def stop(self):
        self._running = False
        self.wait(3000)
    
    def pause(self):
        self._paused = not self._paused
    
    def _cleanup(self):
        print("正在清理资源...")
        try:
            if self._cap:
                self._cap.release()
            if self.pose_backend == 'mediapipe' and self._pose_estimator:
                self._pose_estimator.close()
        except Exception as e:
            print(f"清理错误: {e}")


class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #0d1117; border: 2px solid #30363d; border-radius: 10px; }")
        self.setText("等待视频流...")
        self.setFont(QFont("Microsoft YaHei", 16))
    
    def update_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        except Exception as e:
            print(f"更新帧错误: {e}")
    
    def show_no_signal(self):
        self.clear()
        self.setText("等待视频流...")


class StatusCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        
        layout.addWidget(QLabel(title))
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("color: #e6edf3; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.value_label)
    
    def set_value(self, value, color="#e6edf3"):
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")


class ActionLogWidget(QListWidget):
    ACTION_CONFIG = {
        'walking': {'color': '#3fb950', 'icon': '🚶', 'name': '行走'},
        'running': {'color': '#f0883e', 'icon': '🏃', 'name': '跑步'},
        'sitting': {'color': '#58a6ff', 'icon': '🪑', 'name': '坐下'},
        'standing': {'color': '#a371f7', 'icon': '🧍', 'name': '站立'},
        'raising_hand': {'color': '#39d353', 'icon': '🙋', 'name': '举手'},
        'falling': {'color': '#f85149', 'icon': '⚠️', 'name': '跌倒'}
    }
    
    def add_action(self, action, confidence, timestamp):
        config = self.ACTION_CONFIG.get(action, {'color': '#e6edf3', 'icon': '•', 'name': action})
        text = f"{config['icon']} [{timestamp}] {'⚠️ 警告: ' if action == 'falling' else ''}{config['name']} ({confidence:.1%})"
        item = QListWidgetItem(text)
        item.setForeground(QColor(config['color']))
        self.insertItem(0, item)
        while self.count() > 100:
            self.takeItem(self.count() - 1)


class StatisticsWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        self.stat_labels = {}
        actions = [('walking', '🚶 行走', '#3fb950'), ('running', '🏃 跑步', '#f0883e'), ('sitting', '🪑 坐下', '#58a6ff'), ('standing', '🧍 站立', '#a371f7'), ('raising_hand', '🙋 举手', '#39d353'), ('falling', '⚠️ 跌倒', '#f85149')]
        for i, (key, name, color) in enumerate(actions):
            label = QLabel(f"{name}: 0")
            label.setStyleSheet(f"QLabel {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 15px; color: {color}; }}")
            layout.addWidget(label, i // 2, i % 2)
            self.stat_labels[key] = label
    
    def update_stats(self, action_count):
        names = {'walking': '🚶 行走', 'running': '🏃 跑步', 'sitting': '🪑 坐下', 'standing': '🧍 站立', 'raising_hand': '🙋 举手', 'falling': '⚠️ 跌倒'}
        for key, label in self.stat_labels.items():
            label.setText(f"{names.get(key, key)}: {action_count.get(key, 0)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于单目RGB视频的轻量级人体姿态估计与行为识别系统")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(STYLE_SHEET)
        self.video_thread = None
        self._init_ui()
    
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        header = QLabel("基于单目RGB视频的轻量级人体姿态估计与行为识别系统")
        header.setStyleSheet("background-color: #161b22; border-bottom: 1px solid #30363d; padding: 20px; font-size: 18px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        video_group = QGroupBox("实时视频流")
        video_layout = QVBoxLayout(video_group)
        self.video_widget = VideoWidget()
        video_layout.addWidget(self.video_widget)
        left_layout.addWidget(video_group, stretch=1)
        
        status_group = QGroupBox("系统状态")
        status_layout = QHBoxLayout(status_group)
        self.fps_card = StatusCard("帧率 FPS")
        self.latency_card = StatusCard("处理延迟")
        self.action_card = StatusCard("当前行为")
        self.confidence_card = StatusCard("置信度")
        self.buffer_card = StatusCard("缓冲进度")
        for card in [self.fps_card, self.latency_card, self.action_card, self.confidence_card, self.buffer_card]:
            status_layout.addWidget(card)
        left_layout.addWidget(status_group)
        
        right_panel = QWidget()
        right_panel.setMaximumWidth(380)
        right_layout = QVBoxLayout(right_panel)
        
        config_group = QGroupBox("系统配置")
        config_layout = QGridLayout(config_group)
        config_layout.addWidget(QLabel("视频源:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头 0", "摄像头 1", "视频文件..."])
        config_layout.addWidget(self.source_combo, 0, 1)
        config_layout.addWidget(QLabel("姿态估计:"), 1, 0)
        self.pose_combo = QComboBox()
        self.pose_combo.addItems(["MediaPipe (推荐)", "YOLOv8-Pose"])
        config_layout.addWidget(self.pose_combo, 1, 1)
        config_layout.addWidget(QLabel("行为识别:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["GRU (轻量)", "LSTM", "ST-GCN"])
        config_layout.addWidget(self.model_combo, 2, 1)
        config_layout.addWidget(QLabel("序列长度:"), 3, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, 60)
        self.window_spin.setValue(30)
        config_layout.addWidget(self.window_spin, 3, 1)
        self.smoothing_check = QCheckBox("启用关键点平滑 (1€ Filter)")
        self.smoothing_check.setChecked(True)
        config_layout.addWidget(self.smoothing_check, 4, 0, 1, 2)
        right_layout.addWidget(config_group)
        
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout(control_group)
        self.start_btn = QPushButton("▶ 开始检测")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_btn, 0, 0)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn, 0, 1)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn, 1, 0, 1, 2)
        right_layout.addWidget(control_group)
        
        stats_group = QGroupBox("行为统计")
        stats_layout = QVBoxLayout(stats_group)
        self.statistics_widget = StatisticsWidget()
        stats_layout.addWidget(self.statistics_widget)
        right_layout.addWidget(stats_group)
        
        log_group = QGroupBox("行为日志")
        log_layout = QVBoxLayout(log_group)
        self.action_log = ActionLogWidget()
        log_layout.addWidget(self.action_log)
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.action_log.clear)
        log_layout.addWidget(clear_btn)
        right_layout.addWidget(log_group, stretch=1)
        
        content_layout.addWidget(left_panel, stretch=1)
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)
        
        self.statusBar().showMessage("就绪 | 按 '开始检测' 启动系统")
        self.frame_label = QLabel("帧数: 0")
        self.frame_label.setStyleSheet("color: #8b949e; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.frame_label)
    
    def start_processing(self):
        source_idx = self.source_combo.currentIndex()
        if source_idx == 0:
            source = 0
        elif source_idx == 1:
            source = 1
        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
            if not file_path:
                return
            source = file_path
        
        pose_backend = 'mediapipe' if self.pose_combo.currentIndex() == 0 else 'yolov8'
        model_type = ['gru', 'lstm', 'stgcn'][self.model_combo.currentIndex()]
        
        print(f"准备启动: 视频源={source}, 姿态估计={pose_backend}, 模型={model_type}")
        
        self.video_thread = VideoThread(source=source, pose_backend=pose_backend, model_type=model_type, window_size=self.window_spin.value(), enable_smoothing=self.smoothing_check.isChecked())
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.action_detected.connect(self.on_action_detected)
        self.video_thread.error_occurred.connect(self.on_error)
        self.video_thread.status_update.connect(self.on_status_update)
        self.video_thread.initialized.connect(self.on_initialized)
        
        self.statusBar().showMessage("正在初始化...")
        self.video_thread.start()
    
    def on_initialized(self, success):
        if success:
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("处理中...")
        else:
            self.statusBar().showMessage("初始化失败")
    
    def pause_processing(self):
        if self.video_thread:
            self.video_thread.pause()
            if self.pause_btn.text() == "⏸ 暂停":
                self.pause_btn.setText("▶ 继续")
                self.statusBar().showMessage("已暂停")
            else:
                self.pause_btn.setText("⏸ 暂停")
                self.statusBar().showMessage("处理中...")
    
    def stop_processing(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.video_widget.show_no_signal()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("已停止")
        for card in [self.fps_card, self.latency_card, self.action_card, self.confidence_card, self.buffer_card]:
            card.set_value("--")
    
    def on_frame_ready(self, frame, action, confidence, fps, latency):
        self.video_widget.update_frame(frame)
        self.fps_card.set_value(f"{fps:.1f}", "#3fb950" if fps > 20 else "#f0883e")
        self.latency_card.set_value(f"{latency*1000:.0f}ms", "#3fb950" if latency < 0.05 else "#f0883e")
        
        action_names = {'walking': '行走', 'running': '跑步', 'sitting': '坐下', 'standing': '站立', 'raising_hand': '举手', 'falling': '跌倒', 'no_pose': '无检测', 'collecting': '采集中'}
        action_text = action_names.get(action, action)
        
        if action == 'falling':
            self.action_card.set_value(f"⚠️ {action_text}", "#f85149")
        elif action in action_names:
            self.action_card.set_value(action_text, "#3fb950")
        else:
            self.action_card.set_value(action_text)
        
        self.confidence_card.set_value(f"{confidence:.1%}", "#58a6ff" if confidence > 0.7 else "#8b949e")
    
    def on_action_detected(self, action, confidence, timestamp):
        self.action_log.add_action(action, confidence, timestamp)
    
    def on_status_update(self, status):
        self.frame_label.setText(f"帧数: {status['frame_count']}")
        self.buffer_card.set_value(f"{status['buffer_fill']:.0f}%")
        self.statistics_widget.update_stats(status['action_count'])
    
    def on_error(self, message):
        print(f"错误: {message}")
        QMessageBox.critical(self, "错误", message)
        self.stop_processing()
    
    def closeEvent(self, event):
        self.stop_processing()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
