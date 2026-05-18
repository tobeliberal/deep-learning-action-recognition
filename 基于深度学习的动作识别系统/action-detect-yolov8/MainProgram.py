# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView
import sys
import os
from PIL import ImageFont
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
import torch
from pose_tracker import PoseTracker
from action_recognizer import ActionRecognizer


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        self.conf = 0.25
        self.iou = 0.7

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initMain()
        self.signalconnect()

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.ResetBtn.clicked.connect(self.reset_state)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480
        self.org_path = None
        self.is_camera_open = False
        self.cap = None
        self.persons_data = []
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        self.pose_tracker = PoseTracker(conf=self.conf, iou=self.iou)
        self.action_recognizer = ActionRecognizer()

        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)
        self.colors = tools.Colors()

        self.timer_camera = QTimer()
        self.timer_save_video = QTimer()

        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(36)
        self.ui.tableWidget.setColumnWidth(0, 60)
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 150)
        self.ui.tableWidget.setColumnWidth(3, 80)
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.tableWidget.verticalHeader().setVisible(False)
        self.ui.tableWidget.setAlternatingRowColors(True)

    def _draw_results(self, img, persons_data):
        now_img = img.copy()
        for person in persons_data:
            bbox = person['bbox']
            keypoints = person['keypoints']
            kpts_conf = person['keypoints_conf']
            track_id = person.get('track_id', -1)
            action_result = person.get('action_result', None)

            now_img = tools.draw_skeleton(now_img, keypoints, kpts_conf,
                                          Config.keypoint_conf_threshold)

            if action_result and action_result.get('action_id', -1) >= 0:
                color = self.colors(action_result['action_id'], True)
                now_img = tools.draw_action_label(
                    now_img, bbox, track_id,
                    action_result['action'],
                    action_result['confidence'],
                    color)
            else:
                cv2.rectangle(now_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return now_img

    def _update_status_cards(self, persons_data, process_time=0):
        target_nums = len(persons_data)
        self.ui.nums_card.set_value(str(target_nums),
                                    "#3fb950" if target_nums > 0 else "#e6edf3")

        if process_time > 0:
            fps = 1.0 / max(process_time, 0.001)
            self.ui.fps_card.set_value(f"{fps:.1f}",
                                       "#3fb950" if fps > 20 else "#f0883e")
            self.ui.latency_card.set_value(f"{process_time * 1000:.0f}ms",
                                           "#3fb950" if process_time < 0.05 else "#f0883e")

        if target_nums >= 1:
            person = persons_data[0]
            action_result = person.get('action_result', None)
            if action_result:
                action_text = action_result['action']
                if action_result.get('action_id') == 1:
                    action_text = f"⚠️ {action_text}"
                    self.ui.action_card.set_value(action_text, "#f85149")
                else:
                    self.ui.action_card.set_value(action_text, "#3fb950")
                self.ui.confidence_card.set_value(
                    f"{action_result['confidence']:.1%}",
                    "#58a6ff" if action_result['confidence'] > 0.7 else "#8b949e")
            else:
                self.ui.action_card.set_value("--")
                self.ui.confidence_card.set_value("--")
        else:
            self.ui.action_card.set_value("无检测", "#8b949e")
            self.ui.confidence_card.set_value("--")

    def _update_info_panel(self, persons_data):
        target_nums = len(persons_data)
        self.ui.label_nums.setText(str(target_nums))

        if target_nums >= 1:
            person = persons_data[0]
            action_result = person.get('action_result', None)
            if action_result:
                self.ui.type_lb.setText(action_result['action'])
                self.ui.label_conf.setText(f"{action_result['confidence']:.1%}")
            else:
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')

            bbox = person['bbox']
            self.ui.label_xmin.setText(str(bbox[0]))
            self.ui.label_ymin.setText(str(bbox[1]))
            self.ui.label_xmax.setText(str(bbox[2]))
            self.ui.label_ymax.setText(str(bbox[3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        choose_list = ['全部']
        for i, person in enumerate(persons_data):
            track_id = person.get('track_id', -1)
            action_result = person.get('action_result', None)
            action_name = action_result['action'] if action_result else '未知'
            choose_list.append(f"ID:{track_id}_{action_name}_{i}")

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

    def _update_table(self, persons_data, path=None):
        for person in persons_data:
            row_count = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.insertRow(row_count)

            item_id = QTableWidgetItem(str(row_count + 1))
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_path = QTableWidgetItem(str(path))

            action_result = person.get('action_result', None)
            track_id = person.get('track_id', -1)
            action_name = action_result['action'] if action_result else '未知'
            item_cls = QTableWidgetItem(f"ID:{track_id} {action_name}")
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            conf_str = f"{action_result['confidence']:.1%}" if action_result else ''
            item_conf = QTableWidgetItem(conf_str)
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_location = QTableWidgetItem(str(person['bbox']))

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)

        self.ui.tableWidget.scrollToBottom()

    def open_img(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './',
                                                    "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        t1 = time.time()
        persons = self.pose_tracker.detect(self.org_img)
        t2 = time.time()
        process_time = t2 - t1
        take_time_str = '{:.3f} s'.format(process_time)
        self.ui.time_lb.setText(take_time_str)

        results_data = []
        for person in persons:
            action_result = self.action_recognizer._heuristic_classify(
                person['keypoints'], person['bbox'])
            person['action_result'] = action_result
            results_data.append(person)

        self.persons_data = results_data

        now_img = self._draw_results(self.org_img, results_data)
        self.draw_img = now_img

        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        self.ui.PiclineEdit.setText(self.org_path)

        self._update_status_cards(results_data, process_time)
        self._update_info_panel(results_data)
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self._update_table(results_data, path=self.org_path)

    def combox_change(self):
        com_text = self.ui.comboBox.currentText()
        if not hasattr(self, 'persons_data') or not self.persons_data:
            return

        if com_text == '全部':
            cur_img = self._draw_results(self.org_img, self.persons_data)
            person = self.persons_data[0]
        else:
            try:
                index = int(com_text.split('_')[-1])
                if index < len(self.persons_data):
                    person = self.persons_data[index]
                    cur_img = self._draw_results(self.org_img, [person])
                else:
                    return
            except (ValueError, IndexError):
                return

        action_result = person.get('action_result', None)
        if action_result:
            self.ui.type_lb.setText(action_result['action'])
            self.ui.label_conf.setText(f"{action_result['confidence']:.1%}")
        bbox = person['bbox']
        self.ui.label_xmin.setText(str(bbox[0]))
        self.ui.label_ymin.setText(str(bbox[1]))
        self.ui.label_xmax.setText(str(bbox[2]))
        self.ui.label_ymax.setText(str(bbox[3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './',
                                                    "Image files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.ui.comboBox.clear()
        self.pose_tracker.reset_tracker()
        self.action_recognizer.reset()

        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            t1 = time.time()
            persons = self.pose_tracker.track(now_img, persist=True)
            t2 = time.time()
            process_time = t2 - t1
            take_time_str = '{:.3f} s'.format(process_time)
            self.ui.time_lb.setText(take_time_str)

            active_track_ids = []
            results_data = []
            for person in persons:
                track_id = person['track_id']
                if track_id >= 0:
                    active_track_ids.append(track_id)

                action_result = self.action_recognizer.update(
                    track_id, person['keypoints'], person['bbox'])
                person['action_result'] = action_result
                results_data.append(person)

            self.action_recognizer.remove_stale_tracks(active_track_ids)

            self.persons_data = results_data

            now_img = self._draw_results(now_img, results_data)

            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            self._update_status_cards(results_data, process_time)
            self._update_info_panel(results_data)
            self.ui.tableWidget.setRowCount(0)
            self.ui.tableWidget.clearContents()
            self._update_table(results_data, path=self.org_path)
        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.ui.label_show.setText('等待视频流...')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def reset_state(self):
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.is_camera_open:
            self.is_camera_open = False
            self.timer_camera.stop()

        self.pose_tracker.reset_tracker()
        self.action_recognizer.reset()

        self.org_path = None
        self.persons_data = []
        self.draw_img = None

        self.ui.label_show.clear()
        self.ui.label_show.setText("等待视频流...")
        self.ui.PiclineEdit.clear()
        self.ui.VideolineEdit.clear()
        self.ui.CaplineEdit.setText('摄像头未开启')

        self.ui.label_nums.setText('--')
        self.ui.type_lb.setText('')
        self.ui.label_conf.setText('')
        self.ui.time_lb.setText('--')
        self.ui.label_xmin.setText('')
        self.ui.label_ymin.setText('')
        self.ui.label_xmax.setText('')
        self.ui.label_ymax.setText('')

        self.ui.fps_card.set_value("--")
        self.ui.latency_card.set_value("--")
        self.ui.action_card.set_value("--")
        self.ui.confidence_card.set_value("--")
        self.ui.nums_card.set_value("--")

        self.ui.comboBox.clear()
        self.ui.comboBox.setDisabled(False)
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        self.ui.statusbar.showMessage("已重置 | 点击按钮开始检测")

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width, depth = _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示',
                                          '保存视频检测结果可能需要较长时间，请确认是否继续保存？',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                self.btn2Thread_object = btn2Thread(
                    self.org_path, self.pose_tracker, self.action_recognizer,
                    self.conf, self.iou)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
        else:
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name, end_name = fileName.rsplit(".", 1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                cv2.imwrite(save_img_path, self.draw_img)
                QMessageBox.about(self, '提示', f'图片保存成功!\n文件路径:{save_img_path}')

    def update_process_bar(self, cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', f'视频保存成功!\n文件在{Config.save_path}目录下')
            return
        if self.progress_bar.isVisible() is False:
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total * 100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()


class btn2Thread(QThread):
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, pose_tracker, action_recognizer, conf, iou):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.pose_tracker = pose_tracker
        self.action_recognizer = action_recognizer
        self.conf = conf
        self.iou = iou
        self.colors = tools.Colors()
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.org_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        cur_num = 0

        self.pose_tracker.reset_tracker()
        self.action_recognizer.reset()

        while cap.isOpened() and self.is_running:
            cur_num += 1
            ret, frame = cap.read()
            if ret:
                persons = self.pose_tracker.track(frame, persist=True)

                active_track_ids = []
                results_data = []
                for person in persons:
                    track_id = person['track_id']
                    if track_id >= 0:
                        active_track_ids.append(track_id)
                    action_result = self.action_recognizer.update(
                        track_id, person['keypoints'], person['bbox'])
                    person['action_result'] = action_result
                    results_data.append(person)

                self.action_recognizer.remove_stale_tracks(active_track_ids)

                frame = self._draw_results_save(frame, results_data)
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break

        cap.release()
        out.release()

    def _draw_results_save(self, img, persons_data):
        now_img = img.copy()
        for person in persons_data:
            bbox = person['bbox']
            keypoints = person['keypoints']
            kpts_conf = person['keypoints_conf']
            track_id = person.get('track_id', -1)
            action_result = person.get('action_result', None)

            now_img = tools.draw_skeleton(now_img, keypoints, kpts_conf,
                                          Config.keypoint_conf_threshold)
            if action_result and action_result.get('action_id', -1) >= 0:
                color = self.colors(action_result['action_id'], True)
                now_img = tools.draw_action_label(
                    now_img, bbox, track_id,
                    action_result['action'],
                    action_result['confidence'],
                    color)
            else:
                cv2.rectangle(now_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return now_img

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
