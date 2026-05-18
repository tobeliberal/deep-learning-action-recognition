# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


STYLE_SHEET = """
QMainWindow { background-color: #0d1117; }
QWidget { font-family: 'Microsoft YaHei', sans-serif; color: #e6edf3; }
QGroupBox { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-top: 12px; padding: 15px; font-weight: bold; font-size: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; color: #58a6ff; }
QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 8px 16px; font-size: 13px; color: #e6edf3; min-height: 28px; }
QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
QPushButton:disabled { background-color: #161b22; color: #484f58; }
QPushButton#startBtn { background-color: #238636; border-color: #238636; color: white; }
QPushButton#startBtn:hover { background-color: #2ea043; }
QPushButton#stopBtn { background-color: #da3633; border-color: #da3633; color: white; }
QPushButton#stopBtn:hover { background-color: #f85149; }
QPushButton#saveBtn { background-color: #1f6feb; border-color: #1f6feb; color: white; }
QPushButton#saveBtn:hover { background-color: #388bfd; }
QComboBox { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; color: #e6edf3; min-height: 20px; }
QComboBox:hover { border-color: #58a6ff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #8b949e; margin-right: 10px; }
QComboBox QAbstractItemView { background-color: #161b22; border: 1px solid #30363d; selection-background-color: #1f6feb; color: #e6edf3; }
QLineEdit { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; color: #e6edf3; font-size: 13px; }
QLineEdit:focus { border-color: #58a6ff; }
QTableWidget { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; gridline-color: #21262d; color: #e6edf3; font-size: 13px; }
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected { background-color: #1f6feb; }
QHeaderView::section { background-color: #161b22; border: 1px solid #30363d; padding: 8px; font-weight: bold; color: #58a6ff; }
QStatusBar { background-color: #161b22; border-top: 1px solid #30363d; font-size: 12px; color: #8b949e; }
QLabel#headerTitle { font-size: 20px; font-weight: bold; color: #e6edf3; }
QLabel#valueLabel { color: #58a6ff; font-size: 16px; font-weight: bold; }
QLabel#valueRed { color: #f85149; font-size: 16px; font-weight: bold; }
QLabel#valueGreen { color: #3fb950; font-size: 16px; font-weight: bold; }
QLabel#fieldLabel { color: #8b949e; font-size: 13px; }
"""


class StatusCard(QtWidgets.QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(2)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(title_label)

        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setStyleSheet("color: #e6edf3; font-size: 20px; font-weight: bold;")
        layout.addWidget(self.value_label)

    def set_value(self, value, color="#e6edf3"):
        self.value_label.setText(str(value))
        self.value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        MainWindow.setMinimumSize(QtCore.QSize(1200, 800))
        MainWindow.setStyleSheet(STYLE_SHEET)
        MainWindow.setWindowTitle("基于深度学习的动作识别系统")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QtWidgets.QLabel("基于深度学习的动作识别系统")
        header.setObjectName("headerTitle")
        header.setStyleSheet("background-color: #161b22; border-bottom: 1px solid #30363d; padding: 18px; font-size: 20px; font-weight: bold; color: #e6edf3;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(15)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        video_group = QtWidgets.QGroupBox("实时视频流")
        video_layout = QtWidgets.QVBoxLayout(video_group)
        self.label_show = QtWidgets.QLabel()
        self.label_show.setMinimumSize(770, 480)
        self.label_show.setAlignment(Qt.AlignCenter)
        self.label_show.setStyleSheet("QLabel { background-color: #0d1117; border: 2px solid #30363d; border-radius: 8px; color: #484f58; font-size: 16px; }")
        self.label_show.setText("等待视频流...")
        video_layout.addWidget(self.label_show)
        left_layout.addWidget(video_group, stretch=1)

        status_group = QtWidgets.QGroupBox("系统状态")
        status_layout = QtWidgets.QHBoxLayout(status_group)
        self.fps_card = StatusCard("帧率 FPS")
        self.latency_card = StatusCard("处理延迟")
        self.action_card = StatusCard("当前行为")
        self.confidence_card = StatusCard("置信度")
        self.nums_card = StatusCard("检测人数")
        for card in [self.fps_card, self.latency_card, self.action_card, self.confidence_card, self.nums_card]:
            status_layout.addWidget(card)
        left_layout.addWidget(status_group)

        result_group = QtWidgets.QGroupBox("检测结果与位置信息")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setHorizontalHeaderLabels(["序号", "文件路径", "类别", "置信度", "坐标位置"])
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.tableWidget)
        left_layout.addWidget(result_group)

        right_panel = QtWidgets.QWidget()
        right_panel.setMaximumWidth(380)
        right_panel.setMinimumWidth(340)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        source_group = QtWidgets.QGroupBox("文件导入")
        source_layout = QtWidgets.QGridLayout(source_group)

        self.PicBtn = QtWidgets.QPushButton("📷 图片")
        self.PicBtn.setObjectName("startBtn")
        source_layout.addWidget(self.PicBtn, 0, 0)

        self.PiclineEdit = QtWidgets.QLineEdit()
        self.PiclineEdit.setPlaceholderText("请选择图片文件")
        source_layout.addWidget(self.PiclineEdit, 0, 1)

        self.VideoBtn = QtWidgets.QPushButton("🎬 视频")
        source_layout.addWidget(self.VideoBtn, 1, 0)

        self.VideolineEdit = QtWidgets.QLineEdit()
        self.VideolineEdit.setPlaceholderText("请选择视频文件")
        source_layout.addWidget(self.VideolineEdit, 1, 1)

        self.CapBtn = QtWidgets.QPushButton("🎥 摄像头")
        self.CapBtn.setObjectName("stopBtn")
        source_layout.addWidget(self.CapBtn, 2, 0)

        self.CaplineEdit = QtWidgets.QLineEdit()
        self.CaplineEdit.setPlaceholderText("摄像头未开启")
        self.CaplineEdit.setReadOnly(True)
        source_layout.addWidget(self.CaplineEdit, 2, 1)

        right_layout.addWidget(source_group)

        info_group = QtWidgets.QGroupBox("检测结果")
        info_layout = QtWidgets.QGridLayout(info_group)

        info_layout.addWidget(QtWidgets.QLabel("人员总数:"), 0, 0)
        self.label_nums = QtWidgets.QLabel("--")
        self.label_nums.setObjectName("valueRed")
        self.label_nums.setStyleSheet("color: #f85149; font-size: 18px; font-weight: bold;")
        info_layout.addWidget(self.label_nums, 0, 1)

        info_layout.addWidget(QtWidgets.QLabel("用时:"), 0, 2)
        self.time_lb = QtWidgets.QLabel("--")
        self.time_lb.setObjectName("valueLabel")
        self.time_lb.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
        info_layout.addWidget(self.time_lb, 0, 3)

        info_layout.addWidget(QtWidgets.QLabel("类型:"), 1, 0)
        self.type_lb = QtWidgets.QLabel("--")
        self.type_lb.setStyleSheet("color: #3fb950; font-size: 18px; font-weight: bold;")
        info_layout.addWidget(self.type_lb, 1, 1)

        info_layout.addWidget(QtWidgets.QLabel("置信度:"), 1, 2)
        self.label_conf = QtWidgets.QLabel("--")
        self.label_conf.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
        info_layout.addWidget(self.label_conf, 1, 3)

        info_layout.addWidget(QtWidgets.QLabel("xmin:"), 2, 0)
        self.label_xmin = QtWidgets.QLabel("--")
        self.label_xmin.setStyleSheet("color: #f0883e; font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.label_xmin, 2, 1)

        info_layout.addWidget(QtWidgets.QLabel("ymin:"), 2, 2)
        self.label_ymin = QtWidgets.QLabel("--")
        self.label_ymin.setStyleSheet("color: #f0883e; font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.label_ymin, 2, 3)

        info_layout.addWidget(QtWidgets.QLabel("xmax:"), 3, 0)
        self.label_xmax = QtWidgets.QLabel("--")
        self.label_xmax.setStyleSheet("color: #f0883e; font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.label_xmax, 3, 1)

        info_layout.addWidget(QtWidgets.QLabel("ymax:"), 3, 2)
        self.label_ymax = QtWidgets.QLabel("--")
        self.label_ymax.setStyleSheet("color: #f0883e; font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.label_ymax, 3, 3)

        info_layout.addWidget(QtWidgets.QLabel("目标选择:"), 4, 0)
        self.comboBox = QtWidgets.QComboBox()
        info_layout.addWidget(self.comboBox, 4, 1, 1, 3)

        right_layout.addWidget(info_group)

        control_group = QtWidgets.QGroupBox("操作")
        control_layout = QtWidgets.QHBoxLayout(control_group)
        self.ResetBtn = QtWidgets.QPushButton("🔄 重置")
        self.ResetBtn.setObjectName("startBtn")
        control_layout.addWidget(self.ResetBtn)
        self.SaveBtn = QtWidgets.QPushButton("💾 保存结果")
        self.SaveBtn.setObjectName("saveBtn")
        control_layout.addWidget(self.SaveBtn)
        self.ExitBtn = QtWidgets.QPushButton("🚪 退出系统")
        self.ExitBtn.setObjectName("stopBtn")
        control_layout.addWidget(self.ExitBtn)
        right_layout.addWidget(control_group)

        right_layout.addStretch()

        content_layout.addWidget(left_panel, stretch=1)
        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪 | 点击按钮开始检测")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于深度学习的动作识别系统"))
