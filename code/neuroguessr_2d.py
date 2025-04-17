import os
import sys
import random
import json
import time
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QStackedWidget, QSlider, QMessageBox,
                             QButtonGroup, QGridLayout, QCheckBox, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette, QImage, QFontDatabase

def get_resource_path(relative_path):
    """Get the absolute path to a resource, works for both development and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_path, relative_path)

class BrainSliceView(QLabel):
    """Widget to display a single brain slice with click, drag, and zoom functionality."""
    slice_clicked = pyqtSignal(int, int, int)  # x, y, plane_index
    slice_changed = pyqtSignal(int, int)       # plane_index, delta

    def __init__(self, plane_index, parent=None):
        super().__init__(parent)
        self.plane_index = plane_index
        self.crosshair_pos = (0, 0)
        self.zoom_factor = 1.5
        self.setMinimumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.slice_data = None
        self.template_data = None
        self.plane_names = ["Axial", "Coronal", "Sagittal"]
        self.original_pixmap = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.dragging = False
        self.last_mouse_pos = None
        self.blinking = False
        self.blink_state = True
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink)
        
        self.title = QLabel(self.plane_names[plane_index])
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 0);")
        self.title.setFont(QFont("Helvetica [Cronyx]", 12, QFont.Bold))

    def start_blinking(self):
        self.blinking = True
        self.blink_timer.start(500)

    def stop_blinking(self):
        self.blinking = False
        self.blink_timer.stop()
        self.blink_state = True
        self.update_slice(self.slice_data, self.template_data, self.colormap, self.highlight_region, self.show_atlas)

    def toggle_blink(self):
        self.blink_state = not self.blink_state
        self.update_slice(self.slice_data, self.template_data, self.colormap, self.highlight_region, self.show_atlas)

    def set_crosshair_3d(self, voxel_x, voxel_y, voxel_z):
        if self.plane_index == 0:
            self.crosshair_pos = (voxel_x, voxel_y)
        elif self.plane_index == 1:
            self.crosshair_pos = (voxel_x, voxel_z)
        elif self.plane_index == 2:
            self.crosshair_pos = (voxel_y, voxel_z)
        self.update()

    def wheelEvent(self, event):
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier:  # Cmd on Mac is mapped to ControlModifier
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.1
            elif delta < 0:
                self.zoom_factor /= 1.1
            self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
            self.update()
        else:
            delta = event.angleDelta().y()
            if delta != 0:
                step = 1 if delta > 0 else -1
                self.slice_changed.emit(self.plane_index, step)
        event.accept()

    def update_slice(self, slice_data, template_slice, colormap=None, highlight_region=None, show_atlas=True):
        self.slice_data = slice_data
        self.template_data = template_slice
        self.colormap = colormap
        self.highlight_region = highlight_region
        self.show_atlas = show_atlas
        if slice_data is None or template_slice is None:
            self.clear()
            return
        norm_template = ((template_slice - template_slice.min()) / 
                        (template_slice.max() - template_slice.min() + 1e-8) * 255).astype(np.uint8)
        h, w = norm_template.shape
        colored_slice = np.zeros((h, w, 3), dtype=np.uint8)
        colored_slice[:, :, 0] = norm_template
        colored_slice[:, :, 1] = norm_template
        colored_slice[:, :, 2] = norm_template
        
        if show_atlas and colormap:
            unique_vals = np.unique(slice_data)
            for val in unique_vals:
                if val in colormap and val > 0:
                    mask = (slice_data == val)
                    color = colormap[val]
                    if self.blinking and val == highlight_region and self.blink_state:
                        colored_slice[mask, 0] = 255
                        colored_slice[mask, 1] = 255
                        colored_slice[mask, 2] = 0
                    else:
                        colored_slice[mask, 0] = (0.5 * colored_slice[mask, 0] + 0.5 * color[0]).astype(np.uint8)
                        colored_slice[mask, 1] = (0.5 * colored_slice[mask, 1] + 0.5 * color[1]).astype(np.uint8)
                        colored_slice[mask, 2] = (0.5 * colored_slice[mask, 2] + 0.5 * color[2]).astype(np.uint8)
        elif self.blinking and highlight_region and self.blink_state:
            mask = (slice_data == highlight_region)
            colored_slice[mask, 0] = 255
            colored_slice[mask, 1] = 255
            colored_slice[mask, 2] = 0
        
        qimg = QImage(colored_slice.data, w, h, w * 3, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.original_pixmap:
            return
        painter = QPainter(self)
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        label_width = self.width()
        label_height = self.height()
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2
        scaled_pixmap = self.original_pixmap.scaled(img_width, img_height, 
                                                  Qt.KeepAspectRatio, 
                                                  Qt.SmoothTransformation)
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        x, y = self.crosshair_pos
        scaled_x = int(x * self.zoom_factor) + x_offset
        scaled_y = int(y * self.zoom_factor) + y_offset
        painter.drawLine(scaled_x, y_offset, scaled_x, y_offset + img_height)
        painter.drawLine(x_offset, scaled_y, x_offset + img_width, scaled_y)
        
        painter.setPen(QColor(255, 0, 0))
        font = QFont("Helvetica [Cronyx]", 12)
        painter.setFont(font)
        if self.plane_index == 0:
            painter.drawText(scaled_x - 30, y_offset - 10, "R")
            painter.drawText(scaled_x + 20, y_offset - 10, "L")
            painter.drawText(x_offset - 30, scaled_y - 10, "A")
            painter.drawText(x_offset - 30, scaled_y + 20, "P")
        elif self.plane_index == 1:
            painter.drawText(scaled_x - 30, y_offset - 10, "R")
            painter.drawText(scaled_x + 20, y_offset - 10, "L")
            painter.drawText(x_offset - 30, scaled_y - 10, "S")
            painter.drawText(x_offset - 30, scaled_y + 20, "I")
        elif self.plane_index == 2:
            painter.drawText(scaled_x - 30, y_offset - 10, "A")
            painter.drawText(scaled_x + 20, y_offset - 10, "P")
            painter.drawText(x_offset - 30, scaled_y - 10, "S")
            painter.drawText(x_offset - 30, scaled_y + 20, "I")
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Helvetica [Cronyx]", 12, QFont.Bold))
        painter.drawText(0, 20, label_width, 20, Qt.AlignCenter, self.plane_names[self.plane_index])
        painter.end()

    def mousePressEvent(self, event):
        if not self.original_pixmap or event.button() != Qt.LeftButton:
            return
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        label_width = self.width()
        label_height = self.height()
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2
        x = event.x() - x_offset
        y = event.y() - y_offset
        orig_x = int(x / self.zoom_factor)
        orig_y = int(y / self.zoom_factor)
        if 0 <= orig_x < self.original_pixmap.width() and 0 <= orig_y < self.original_pixmap.height():
            self.crosshair_pos = (orig_x, orig_y)
            self.update()
            self.slice_clicked.emit(orig_x, orig_y, self.plane_index)
        self.dragging = True
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if not self.original_pixmap or not self.dragging:
            return
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        label_width = self.width()
        label_height = self.height()
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2
        x = event.x() - x_offset
        y = event.y() - y_offset
        orig_x = int(x / self.zoom_factor)
        orig_y = int(y / self.zoom_factor)
        orig_x = max(0, min(orig_x, self.original_pixmap.width() - 1))
        orig_y = max(0, min(orig_y, self.original_pixmap.height() - 1))
        self.crosshair_pos = (orig_x, orig_y)
        self.update()
        self.slice_clicked.emit(orig_x, orig_y, self.plane_index)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

class NeuroGuessrGame(QMainWindow):
    """Main window for the NeuroGuessr game with landing page and three modes."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGuessr")
        self.showMaximized()
        self.score = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.current_target = None
        self.game_running = False
        self.game_mode = "Practice"
        self.time_remaining = 180
        self.correct_guesses = []
        self.incorrect_guesses = []
        self.streak_guessed_regions = []
        self.brain_data = None
        self.template_data = None
        self.region_map = None
        self.colormap = {}
        self.region_info = {}
        self.current_slices = [None, None, None]
        self.current_positions = [0, 0, 0]
        self.selected_position = None
        self.crosshair_3d = (0, 0, 0)
        self.show_atlas = True
        self.all_regions = []
        self.remaining_regions = []
        self.start_time = None
        self.total_time = 0
        self.pr_file = os.path.join(Path.home(), ".neuroguessr", "pr.json")
        self.atlas_options = {
            "AAL": (
                get_resource_path("data/aal_stride_regrid.nii.gz"),
                get_resource_path("data/aal.txt")
            ),
            "Brodmann": (
                get_resource_path("data/brodmann_grid_stride.nii.gz"),
                get_resource_path("data/brodmann.txt")
            ),
            "Harvard Oxford": (
                get_resource_path("data/HarvardOxford-cort-maxprob-thr25-1mm_stride.nii.gz"),
                get_resource_path("data/HarvardOxford-Cortical.txt")
            ),
            "Subcortical": (
                get_resource_path("data/HarvardOxford-sub-maxprob-thr25-1mm_stride.nii.gz"),
                get_resource_path("data/HarvardOxford-Subcortical.txt")
            ),
            "Cerebellum": (
                get_resource_path("data/Cerebellum-MNIfnirt-maxprob-thr25-1mm_stride.nii.gz"),
                get_resource_path("data/Cerebellum_MNIfnirt.txt")
            ),
            "Xtract": (
                get_resource_path("data/xtract_stride.nii.gz"),
                get_resource_path("data/xtract.txt")
            )
        }
        self.pr_data = self.load_pr()
        self.current_atlas = "AAL"
        self.setup_ui()
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_timer)

    def load_pr(self):
        """Load personal records from JSON file."""
        os.makedirs(os.path.dirname(self.pr_file), exist_ok=True)
        try:
            with open(self.pr_file, 'r') as f:
                data = json.load(f)
                for atlas in self.atlas_options.keys():
                    if atlas not in data:
                        data[atlas] = {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0}
                    else:
                        data[atlas].setdefault("time", float("inf"))
                        data[atlas].setdefault("errors", 0)
                        data[atlas].setdefault("best_ratio", 0.0)
                        data[atlas].setdefault("best_streak", 0)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {atlas: {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0}
                    for atlas in self.atlas_options.keys()}

    def save_pr(self):
        """Save personal records to JSON file."""
        try:
            with open(self.pr_file, 'w') as f:
                json.dump(self.pr_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save PR data: {e}")

    def setup_ui(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.landing_widget = QWidget()
        landing_layout = QVBoxLayout(self.landing_widget)
        landing_layout.setAlignment(Qt.AlignCenter)
        landing_layout.setSpacing(30)
        landing_layout.setContentsMargins(50, 50, 50, 50)

        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignCenter)
        
        logo_label = QLabel()
        logo_path = get_resource_path("code/neuroguessr5.png")
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            logo_label.setText("Logo Not Found")
            print(f"Warning: Could not load logo at {logo_path}")
        else:
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        top_layout.addWidget(logo_label)
        
        title_label = QLabel("NeuroGuessr")
        title_label.setFont(QFont("Helvetica [Cronyx]", 70, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        top_layout.addWidget(title_label)
        
        landing_layout.addLayout(top_layout)

        mode_label = QLabel("Game Mode")
        mode_label.setStyleSheet("color: white; font-size: 18px;")
        mode_label.setFont(QFont("Helvetica [Cronyx]", 20, QFont.Bold))
        mode_label.setAlignment(Qt.AlignCenter)
        landing_layout.addWidget(mode_label)
        
        mode_buttons_layout = QHBoxLayout()
        mode_buttons_layout.setSpacing(20)
        
        self.mode_button_group = QButtonGroup(self)
        
        practice_button = QPushButton("Practice")
        practice_button.setStyleSheet("""
            QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
            QPushButton:checked {background-color: #3E3E42; border: 2px solid #0078D7;}
            QPushButton:hover {background-color: #3E3E42; border: 2px solid #0078D7;}
        """)
        practice_button.setFont(QFont("Helvetica [Cronyx]", 16))
        practice_button.setCheckable(True)
        practice_button.setChecked(True)
        self.mode_button_group.addButton(practice_button, 0)
        
        contre_button = QPushButton("Contre la Montre")
        contre_button.setStyleSheet("""
            QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
            QPushButton:checked {background-color: #3E3E42; border: 2px solid #0078D7;}
            QPushButton:hover {background-color: #3E3E42; border: 2px solid #0078D7;}
        """)
        contre_button.setFont(QFont("Helvetica [Cronyx]", 16))
        contre_button.setCheckable(True)
        self.mode_button_group.addButton(contre_button, 1)
        
        streak_button = QPushButton("Streak")
        streak_button.setStyleSheet("""
            QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
            QPushButton:checked {background-color: #3E3E42; border: 2px solid #0078D7;}
            QPushButton:hover {background-color: #3E3E42; border: 2px solid #0078D7;}
        """)
        streak_button.setFont(QFont("Helvetica [Cronyx]", 16))
        streak_button.setCheckable(True)
        self.mode_button_group.addButton(streak_button, 2)
        
        mode_buttons_layout.addWidget(practice_button)
        mode_buttons_layout.addWidget(contre_button)
        mode_buttons_layout.addWidget(streak_button)
        landing_layout.addLayout(mode_buttons_layout)

        atlas_label = QLabel("Select Atlas")
        atlas_label.setStyleSheet("color: white; font-size: 18px;")
        atlas_label.setFont(QFont("Helvetica [Cronyx]", 20, QFont.Bold))
        atlas_label.setAlignment(Qt.AlignCenter)
        landing_layout.addWidget(atlas_label)
        
        atlas_buttons_layout = QGridLayout()
        atlas_buttons_layout.setSpacing(10)
        
        self.atlas_button_group = QButtonGroup(self)
        
        atlas_names = list(self.atlas_options.keys())
        for i, atlas_name in enumerate(atlas_names):
            atlas_button = QPushButton(atlas_name)
            atlas_button.setStyleSheet("""
                QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
                QPushButton:checked {background-color: #3E3E42; border: 2px solid #0078D7;}
                QPushButton:hover {background-color: #3E3E42; border: 2px solid #0078D7;}
            """)
            atlas_button.setFont(QFont("Helvetica [Cronyx]", 16))
            atlas_button.setCheckable(True)
            if atlas_name == self.current_atlas:
                atlas_button.setChecked(True)
            self.atlas_button_group.addButton(atlas_button, i)
            row = i // 3
            col = i % 3
            atlas_buttons_layout.addWidget(atlas_button, row, col)
        
        landing_layout.addLayout(atlas_buttons_layout)
        
        self.pr_box = QGroupBox("Personal Best")
        self.pr_box.setStyleSheet("""
            QGroupBox {color: white; font-size: 18px; font-weight: bold; border: 2px solid #444; border-radius: 10px; padding: 10px;}
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;}
        """)
        self.pr_box.setFont(QFont("Helvetica [Cronyx]",20, QFont.Bold))
        self.pr_box.setAlignment(Qt.AlignCenter)
        pr_layout = QVBoxLayout(self.pr_box)
        pr_layout.setSpacing(20)
        pr_layout.setAlignment(Qt.AlignCenter)
        
        ratio_layout = QHBoxLayout()
        ratio_layout.setAlignment(Qt.AlignCenter)
        ratio_icon = QLabel()
        ratio_icon_path = get_resource_path("code/speedometer.png")
        ratio_pixmap = QPixmap(ratio_icon_path)
        if not ratio_pixmap.isNull():
            ratio_icon.setPixmap(ratio_pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            ratio_icon.setText("⚡")
        ratio_layout.addWidget(ratio_icon)
        self.ratio_pr_label = QLabel("Accuracy: 0")
        self.ratio_pr_label.setStyleSheet("color: white; font-size: 18px;")
        self.ratio_pr_label.setFont(QFont("Helvetica [Cronyx]", 22))
        ratio_layout.addWidget(self.ratio_pr_label)
        ratio_layout.addSpacing(10)
        pr_layout.addLayout(ratio_layout)
        
        time_layout = QHBoxLayout()
        time_layout.setAlignment(Qt.AlignCenter)
        time_icon = QLabel()
        time_icon_path = get_resource_path("code/stopwatch.png")
        time_pixmap = QPixmap(time_icon_path)
        if not time_pixmap.isNull():
            time_icon.setPixmap(time_pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            time_icon.setText("⏱")
        time_layout.addWidget(time_icon)
        self.time_pr_label = QLabel("Time: 0")
        self.time_pr_label.setStyleSheet("color: white; font-size: 18px;")
        self.time_pr_label.setFont(QFont("Helvetica [Cronyx]",22))
        time_layout.addWidget(self.time_pr_label)
        time_layout.addSpacing(10)
        pr_layout.addLayout(time_layout)
        
        streak_layout = QHBoxLayout()
        streak_layout.setAlignment(Qt.AlignCenter)
        pr_layout.addLayout(streak_layout)
        streak_icon = QLabel()
        streak_icon_path = get_resource_path("code/flame.png")
        streak_pixmap = QPixmap(streak_icon_path)
        if not streak_pixmap.isNull():
            streak_icon.setPixmap(streak_pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            streak_icon.setText("🔥")
        streak_layout.addWidget(streak_icon)
        self.streak_pr_label = QLabel("Streak: 0")
        self.streak_pr_label.setStyleSheet("color: white; font-size: 18px;")
        self.streak_pr_label.setFont(QFont("Helvetica [Cronyx]", 22))
        streak_layout.addWidget(self.streak_pr_label)
        streak_layout.addSpacing(10)
        
        landing_layout.addWidget(self.pr_box)
        self.update_pr_label()
        self.atlas_button_group.buttonClicked.connect(self.update_pr_label)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)
        
        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet("""
            QPushButton {font-size: 18px; padding: 15px 40px; background-color: #f44336; color: white; border-radius: 10px; border: none; font-weight: bold;}
            QPushButton:hover {background-color: #d32f2f;}
        """)
        quit_button.setFont(QFont("Helvetica [Cronyx]", 18, QFont.Bold))
        quit_button.clicked.connect(QApplication.instance().quit)

        play_button = QPushButton("Play")
        play_button.setStyleSheet("""
            QPushButton {font-size: 18px; padding: 15px 40px; background-color: #4CAF50; color: white; border-radius: 10px; border: none; font-weight: bold;}
            QPushButton:hover {background-color: #45a049;}
        """)
        play_button.setFont(QFont("Helvetica [Cronyx]", 18, QFont.Bold))
        play_button.clicked.connect(self.start_game_from_landing)
        
        buttons_layout.addWidget(quit_button)
        buttons_layout.addWidget(play_button)
        landing_layout.addLayout(buttons_layout)
        landing_layout.addStretch()

        self.stacked_widget.addWidget(self.landing_widget)

        self.game_widget = QWidget()
        main_game_layout = QHBoxLayout(self.game_widget)

        left_panel = QWidget()
        game_layout = QVBoxLayout(left_panel)

        selection_layout = QHBoxLayout()
        atlas_layout = QHBoxLayout()
        atlas_label = QLabel("Active Atlas:")
        atlas_label.setStyleSheet("color: white;")
        atlas_label.setFont(QFont("Helvetica [Cronyx]", 14))
        self.active_atlas_label = QLabel(self.current_atlas)
        self.active_atlas_label.setStyleSheet("color: white; font-weight: bold;")
        self.active_atlas_label.setFont(QFont("Helvetica [Cronyx]", 14, QFont.Bold))
        atlas_layout.addWidget(atlas_label)
        atlas_layout.addWidget(self.active_atlas_label)
        self.atlas_toggle = QCheckBox("Show Atlas Regions")
        self.atlas_toggle.setChecked(True)
        self.atlas_toggle.setStyleSheet("color: white; font-size: 14px;")
        self.atlas_toggle.setFont(QFont("Helvetica [Cronyx]", 14))
        self.atlas_toggle.stateChanged.connect(self.toggle_atlas_visibility)
        atlas_layout.addWidget(self.atlas_toggle)
        selection_layout.addLayout(atlas_layout)
        selection_layout.addStretch()
        game_layout.addLayout(selection_layout)

        status_layout = QHBoxLayout()
        self.target_label = QLabel("Target: Not Started")
        self.target_label.setFont(QFont("Helvetica [Cronyx]", 24, QFont.Bold))
        self.target_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.timer_label = QLabel("Time: 3:00")
        self.timer_label.setFont(QFont("Helvetica [Cronyx]", 14))
        self.target_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        self.timer_label.setFont(QFont("Helvetica [Cronyx]", 14))
        self.timer_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        score_layout = QVBoxLayout()
        self.score_label = QLabel("Correct: 0")
        self.score_label.setFont(QFont("Helvetica [Cronyx]", 14))
        self.score_label.setStyleSheet("color: white;")
        self.error_label = QLabel("Errors: 0")
        self.error_label.setFont(QFont("Helvetica [Cronyx]", 14))
        self.error_label.setStyleSheet("color: white;")
        score_layout.addWidget(self.score_label)
        score_layout.addWidget(self.error_label)
        status_layout.addWidget(self.target_label, 3)
        status_layout.addWidget(self.timer_label, 1)
        status_layout.addLayout(score_layout, 1)
        game_layout.addLayout(status_layout)

        views_layout = QHBoxLayout()
        self.slice_views = []
        for i in range(3):
            view = BrainSliceView(i)
            view.slice_clicked.connect(self.handle_slice_click)
            view.slice_changed.connect(self.handle_slice_change)
            self.slice_views.append(view)
            views_layout.addWidget(view)
        game_layout.addLayout(views_layout, 1)

        slider_layout = QHBoxLayout()
        z_layout = QVBoxLayout()
        z_label = QLabel("Axial")
        z_label.setAlignment(Qt.AlignCenter)
        z_label.setStyleSheet("color: white;")
        z_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(100)
        self.z_slider.setValue(50)
        self.z_slider.valueChanged.connect(lambda v: self.update_slice_position(0, v))
        z_layout.addWidget(z_label)
        z_layout.addWidget(self.z_slider)
        y_layout = QVBoxLayout()
        y_label = QLabel("Coronal")
        y_label.setAlignment(Qt.AlignCenter)
        y_label.setStyleSheet("color: white;")
        y_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.setMaximum(100)
        self.y_slider.setValue(50)
        self.y_slider.valueChanged.connect(lambda v: self.update_slice_position(1, v))
        y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_slider)
        x_layout = QVBoxLayout()
        x_label = QLabel("Sagittal")
        x_label.setAlignment(Qt.AlignCenter)
        x_label.setStyleSheet("color: white;")
        x_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(100)
        self.x_slider.setValue(50)
        self.x_slider.valueChanged.connect(lambda v: self.update_slice_position(2, v))
        x_layout.addWidget(x_label)
        x_layout.addWidget(self.x_slider)
        slider_layout.addLayout(z_layout)
        slider_layout.addLayout(y_layout)
        slider_layout.addLayout(x_layout)
        game_layout.addLayout(slider_layout)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Game")
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;}
            QPushButton:hover {background-color: #45a049;}
        """)
        self.start_button.setFont(QFont("Helvetica [Cronyx]", 16))
        
        self.guess_button = QPushButton("Confirm Guess")
        self.guess_button.clicked.connect(self.validate_guess)
        self.guess_button.setEnabled(False)
        self.guess_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;}
            QPushButton:disabled {background-color: #cccccc; color: #666666;}
            QPushButton:hover:enabled {background-color: #45a049;}
        """)
        self.guess_button.setFont(QFont("Helvetica [Cronyx]", 16))
        
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        self.help_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #2196F3; color: white; border-radius: 5px;}
            QPushButton:hover {background-color: #0b7dda;}
        """)
        self.help_button.setFont(QFont("Helvetica [Cronyx]", 16))
        
        self.menu_button = QPushButton("Menu")
        self.menu_button.clicked.connect(self.show_menu)
        self.menu_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #FF9800; color: white; border-radius: 5px;}
            QPushButton:hover {background-color: #e68a00;}
        """)
        self.menu_button.setFont(QFont("Helvetica [Cronyx]", 16))
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.guess_button)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.menu_button)
        game_layout.addLayout(button_layout)

        self.guess_button.hide()

        self.memo_widget = QWidget()
        memo_layout = QVBoxLayout(self.memo_widget)
        memo_layout.setContentsMargins(10, 10, 10, 10)
        self.memo_title = QLabel("Region Information")
        self.memo_title.setFont(QFont("Helvetica [Cronyx]", 16, QFont.Bold))
        self.memo_title.setStyleSheet("color: white;")
        self.memo_title.setAlignment(Qt.AlignCenter)
        memo_layout.addWidget(self.memo_title)
        self.memo_text = QTextEdit()
        self.memo_text.setReadOnly(True)
        self.memo_text.setStyleSheet("""
            QTextEdit {background-color: #2D2D30; color: white; border: 1px solid #444; border-radius: 5px; padding: 5px; font-size: 14px;}
        """)
        self.memo_text.setFont(QFont("Helvetica [Cronyx]", 14))
        self.memo_text.setMinimumWidth(300)
        self.memo_text.setMaximumWidth(400)
        memo_layout.addWidget(self.memo_text)
        memo_layout.addStretch()

        main_game_layout.addWidget(left_panel, 3)
        main_game_layout.addWidget(self.memo_widget, 1)

        self.setFocusPolicy(Qt.StrongFocus)
        self.keyPressEvent = self.handle_key_press

        self.stacked_widget.addWidget(self.game_widget)
        self.stacked_widget.setCurrentWidget(self.landing_widget)
        self.load_data()

    def update_pr_label(self):
        atlas_names = list(self.atlas_options.keys())
        selected_atlas_id = self.atlas_button_group.checkedId()
        atlas = atlas_names[selected_atlas_id]
        pr = self.pr_data.get(atlas, {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0})
        
        if pr["time"] == float("inf"):
            self.time_pr_label.setText("0")
        else:
            minutes = pr["time"] // 60
            seconds = pr["time"] % 60
            self.time_pr_label.setText(f"{minutes}'{seconds:02d}\" ")
        
        if pr["best_ratio"] == 0.0:
            self.ratio_pr_label.setText("0")
        else:
            self.ratio_pr_label.setText(f"{pr['best_ratio']:.1f}%")
        
        if pr["best_streak"] == 0:
            self.streak_pr_label.setText("0")
        else:
            self.streak_pr_label.setText(f"{pr['best_streak']}")

    def toggle_atlas_visibility(self, state):
        self.show_atlas = (state == Qt.Checked)
        self.update_all_slices()

    def reset_game_ui(self):
        self.start_button.show()
        self.guess_button.hide()
        self.target_label.setText("Target: Not Started")
        self.score_label.setText("Correct: 0")
        self.error_label.setText("Errors: 0")
        if self.game_mode == "Contre la Montre":
            self.timer_label.setText("Time: 0:00")
        else:
            self.timer_label.setText("Time: N/A")
        self.guess_button.setEnabled(False)
        self.selected_position = None
        self.current_target = None
        self.score = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.correct_guesses = []
        self.incorrect_guesses = []
        self.streak_guessed_regions = []
        self.all_regions = []
        self.remaining_regions = []
        self.start_time = None
        self.total_time = 0
        self.time_remaining = 180
        self.game_running = False
        self.show_atlas = True
        self.atlas_toggle.setChecked(True)
        for view in self.slice_views:
            view.stop_blinking()
        self.memo_widget.setVisible(False)

    def handle_slice_change(self, plane_index, delta):
        if plane_index == 0:
            slider = self.z_slider
        elif plane_index == 1:
            slider = self.y_slider
        else:
            slider = self.x_slider
        new_value = slider.value() + delta
        new_value = max(slider.minimum(), min(slider.maximum(), new_value))
        slider.setValue(new_value)
        self.update_slice_position(plane_index, new_value)

    def start_game_from_landing(self):
        selected_mode_id = self.mode_button_group.checkedId()
        if selected_mode_id == 0:
            self.game_mode = "Practice"
        elif selected_mode_id == 1:
            self.game_mode = "Contre la Montre"
        else:
            self.game_mode = "Streak"
        selected_atlas_id = self.atlas_button_group.checkedId()
        atlas_names = list(self.atlas_options.keys())
        self.current_atlas = atlas_names[selected_atlas_id]
        self.active_atlas_label.setText(self.current_atlas)
        self.load_data()
        self.set_game_mode(self.game_mode)
        self.reset_game_ui()
        self.update_pr_label()
        self.memo_widget.setVisible(self.game_mode == "Practice")
        self.stacked_widget.setCurrentWidget(self.game_widget)

    def handle_key_press(self, event):
        if event.key() == Qt.Key_Space and self.guess_button.isEnabled():
            self.validate_guess()

    def load_data(self):
        atlas_name = self.current_atlas
        atlas_file, region_file = self.atlas_options[atlas_name]
        template_file = get_resource_path("data/MNI_template_1mm_stride.nii.gz")
        json_file = os.path.splitext(region_file)[0] + ".json"
        try:
            if os.path.exists(template_file):
                self.template_data = nib.load(template_file)
            else:
                raise FileNotFoundError(f"Template file {template_file} not found.")
            if os.path.exists(atlas_file) and os.path.exists(region_file):
                self.brain_data = nib.load(atlas_file)
                region_df = pd.read_csv(region_file, sep="\s+", comment="#", header=None,
                                        names=["Index", "RegionName", "R", "G", "B", "A"])
                self.region_map = {row["Index"]: row["RegionName"] for _, row in region_df.iterrows()}
                self.colormap = {row["Index"]: (row["R"], row["G"], row["B"]) for _, row in region_df.iterrows()}
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        self.region_info = json.load(f)
                else:
                    self.region_info = {}
                    print(f"Warning: JSON file {json_file} not found.")
                if self.brain_data.shape != self.template_data.shape:
                    raise ValueError("Atlas and template dimensions do not match.")
                self.z_slider.setMaximum(self.brain_data.shape[2] - 1)
                self.y_slider.setMaximum(self.brain_data.shape[1] - 1)
                self.x_slider.setMaximum(self.brain_data.shape[0] - 1)
                self.z_slider.setValue(self.brain_data.shape[2] // 2)
                self.y_slider.setValue(self.brain_data.shape[1] // 2)
                self.x_slider.setValue(self.brain_data.shape[0] // 2)
                self.update_all_slices()
            else:
                self.load_dummy_data()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}\nUsing dummy data.")
            self.load_dummy_data()

    def load_dummy_data(self):
        dummy_shape = (256, 256, 256)
        dummy_data = np.zeros(dummy_shape, dtype=np.int16)
        dummy_template = np.random.normal(100, 20, dummy_shape).astype(np.float32)
        regions = {
            1: "Left Cerebral Cortex", 2: "Right Cerebral Cortex", 3: "Left Hippocampus",
            4: "Right Hippocampus", 5: "Left Thalamus", 6: "Right Thalamus",
            7: "Left Amygdala", 8: "Right Amygdala", 9: "Left Caudate", 10: "Right Caudate"
        }
        self.region_info = {
            "1": {"name": "Left Cerebral Cortex", "structure": ["Dummy structure info for cortex."], "function": ["Dummy function info for cortex."]},
            "2": {"name": "Right Cerebral Cortex", "structure": ["Dummy structure info for cortex."], "function": ["Dummy function info for cortex."]}
        }
        for i in range(1, 11):
            self.colormap[i] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            center = np.array([random.randint(30, 226), random.randint(30, 226), random.randint(30, 226)])
            size = random.randint(10, 20)
            if i % 2 == 0:
                center[0] = random.randint(130, 226)
            else:
                center[0] = random.randint(30, 126)
            for x in range(max(0, center[0]-size), min(dummy_shape[0], center[0]+size)):
                for y in range(max(0, center[1]-size), min(dummy_shape[1], center[1]+size)):
                    for z in range(max(0, center[2]-size), min(dummy_shape[2], center[2]+size)):
                        if ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2) < size**2:
                            dummy_data[x, y, z] = i
        affine = np.eye(4)
        self.brain_data = nib.Nifti1Image(dummy_data, affine)
        self.template_data = nib.Nifti1Image(dummy_template, affine)
        self.region_map = regions
        self.z_slider.setMaximum(dummy_shape[2] - 1)
        self.y_slider.setMaximum(dummy_shape[1] - 1)
        self.x_slider.setMaximum(dummy_shape[0] - 1)
        self.z_slider.setValue(dummy_shape[2] // 2)
        self.y_slider.setValue(dummy_shape[1] // 2)
        self.x_slider.setValue(dummy_shape[0] // 2)
        self.update_all_slices()

    def set_game_mode(self, mode):
        self.game_mode = mode
        if mode == "Practice":
            self.timer_label.setText("Time: N/A")
            self.score_label.setText("Correct: 0")
            self.error_label.setText("Errors: 0")
            self.memo_widget.setVisible(True)
        elif mode == "Contre la Montre":
            self.timer_label.setText("Time: 0'00\" ")
            self.score_label.setText("Regions Found: 0")
            self.memo_widget.setVisible(False)
        else:
            self.timer_label.setText("Time: N/A")
            self.score_label.setText("Streak: 0")
            self.memo_widget.setVisible(False)

    def start_game(self):
        self.score = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.correct_guesses = []
        self.incorrect_guesses = []
        self.streak_guessed_regions = []
        self.game_running = True
        if self.game_mode == "Contre la Montre":
            self.all_regions = [int(val) for val in np.unique(self.brain_data.get_fdata().astype(np.int32))
                               if val > 0 and val in self.region_map]
            self.remaining_regions = self.all_regions.copy()
            random.shuffle(self.remaining_regions)
            self.start_time = time.time()
            self.score_label.setText(f"Regions Found: 0/{len(self.all_regions)}")
            self.timer_label.setText("Time: 0'00\" ")
        else:
            self.time_remaining = 180 if self.game_mode == "Contre la Montre" else 0
            self.score_label.setText(f"{'Streak' if self.game_mode == 'Streak' else 'Correct'}: {self.score}")
            self.timer_label.setText("Time: N/A")
        self.error_label.setText("Errors: 0")
        self.start_button.hide()
        self.guess_button.show()
        self.menu_button.show()
        self.guess_button.setEnabled(False)
        self.memo_widget.setVisible(self.game_mode == "Practice")
        self.select_new_target()
        if self.game_mode == "Contre la Montre":
            self.game_timer.start(1000)

    def update_timer_display(self):
        minutes = self.time_remaining // 60
        seconds = self.time_remaining % 60
        self.timer_label.setText(f"Time: {minutes}:{seconds:02d}")

    def update_memo_content(self):
        if not self.current_target or self.game_mode != "Practice":
            self.memo_text.setText("")
            return
        region_id = str(self.current_target)
        if region_id in self.region_info:
            info = self.region_info[region_id]
            content = f"""
            <h2 style='margin-bottom: 15px;'>{info['name']}</h2>
            <h3 style='margin-bottom: 10px;'>Structure:</h3>
            <ul style='line-height: 1.8; margin-bottom: 20px;'>
            """
            for item in info.get('structure', []):
                content += f"<li>{item}</li>"
            content += """
            </ul>
            <h3 style='margin-bottom: 10px;'>Function:</h3>
            <ul style='line-height: 1.8;'>
            """
            for item in info.get('function', []):
                content += f"<li>{item}</li>"
            content += "</ul>"
            self.memo_text.setHtml(content)
        else:
            self.memo_text.setText("No information available for this region.")

    def update_slice_position(self, plane_index, value):
        self.current_positions[plane_index] = value
        if plane_index == 0:
            self.crosshair_3d = (self.crosshair_3d[0], self.crosshair_3d[1], value)
        elif plane_index == 1:
            self.crosshair_3d = (self.crosshair_3d[0], value, self.crosshair_3d[2])
        elif plane_index == 2:
            self.crosshair_3d = (value, self.crosshair_3d[1], self.crosshair_3d[2])
        self.update_all_slices()

    def update_all_slices(self):
        if self.brain_data is None or self.template_data is None:
            return
        brain_3d = self.brain_data.get_fdata().astype(np.int32)
        template_3d = self.template_data.get_fdata()
        z, y, x = self.current_positions
        brain_shape = self.brain_data.shape
        x = min(max(x, 0), brain_shape[0] - 1)
        y = min(max(y, 0), brain_shape[1] - 1)
        z = min(max(z, 0), brain_shape[2] - 1)
        axial_slice = brain_3d[:, :, z].T
        coronal_slice = brain_3d[:, y, :].T
        sagittal_slice = brain_3d[x, :, :].T
        axial_template = template_3d[:, :, z].T
        coronal_template = template_3d[:, y, :].T
        sagittal_template = template_3d[x, :, :].T
        highlight_region = self.current_target if self.consecutive_errors >= 3 and self.game_mode == "Practice" else None
        self.slice_views[0].update_slice(axial_slice, axial_template, self.colormap, highlight_region, self.show_atlas)
        self.slice_views[1].update_slice(coronal_slice, coronal_template, self.colormap, highlight_region, self.show_atlas)
        self.slice_views[2].update_slice(sagittal_slice, sagittal_template, self.colormap, highlight_region, self.show_atlas)
        voxel_x, voxel_y, voxel_z = self.crosshair_3d
        for view in self.slice_views:
            view.set_crosshair_3d(voxel_x, voxel_y, voxel_z)

    def select_new_target(self):
        if self.game_mode == "Contre la Montre":
            if not self.remaining_regions:
                self.end_game()
                return
            self.current_target = self.remaining_regions.pop(0)
        else:
            valid_regions = [int(val) for val in np.unique(self.brain_data.get_fdata().astype(np.int32))
                            if val > 0 and val in self.region_map]
            if not valid_regions:
                QMessageBox.warning(self, "Error", "No valid regions found.")
                return
            if self.game_mode == "Streak" and self.streak_guessed_regions:
                weights = [0.2 if region in self.streak_guessed_regions else 1.0 for region in valid_regions]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                region_id = np.random.choice(valid_regions, p=weights)
            else:
                region_id = random.choice(valid_regions)
            self.current_target = region_id
        self.target_label.setText(f"Find: {self.region_map[self.current_target]}")
        self.consecutive_errors = 0
        for view in self.slice_views:
            view.stop_blinking()
        self.update_memo_content()

    def handle_slice_click(self, x, y, plane_index):
        if not self.game_running or not self.brain_data:
            return
        brain_shape = self.brain_data.shape
        if plane_index == 0:
            voxel_x = min(max(x, 0), brain_shape[0] - 1)
            voxel_y = min(max(y, 0), brain_shape[1] - 1)
            voxel_z = self.current_positions[0]
        elif plane_index == 1:
            voxel_x = min(max(x, 0), brain_shape[0] - 1)
            voxel_y = self.current_positions[1]
            voxel_z = min(max(y, 0), brain_shape[2] - 1)
        else:
            voxel_x = self.current_positions[2]
            voxel_y = min(max(x, 0), brain_shape[1] - 1)
            voxel_z = min(max(y, 0), brain_shape[2] - 1)
        self.current_positions = [voxel_z, voxel_y, voxel_x]
        self.z_slider.setValue(voxel_z)
        self.y_slider.setValue(voxel_y)
        self.x_slider.setValue(voxel_x)
        self.selected_position = (voxel_x, voxel_y, voxel_z)
        self.crosshair_3d = (voxel_x, voxel_y, voxel_z)
        self.update_all_slices()
        self.guess_button.setEnabled(True)
        self.guess_button.setText("Confirm Guess")
        self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; font-weight: bold;")

    def validate_guess(self):
        if not self.selected_position or not self.game_running:
            return
        voxel_x, voxel_y, voxel_z = self.selected_position
        brain_data = self.brain_data.get_fdata().astype(np.int32)
        try:
            clicked_region = int(brain_data[voxel_x, voxel_y, voxel_z])
        except IndexError:
            clicked_region = 0
        target_name = self.region_map.get(self.current_target, "Unknown")
        clicked_name = self.region_map.get(clicked_region, "Background/Unknown")
        
        if clicked_region == self.current_target:
            self.score += 1
            self.correct_guesses.append(target_name)
            if self.game_mode == "Practice":
                self.score_label.setText(f"Correct: {self.score}")
            elif self.game_mode == "Streak":
                self.streak_guessed_regions.append(self.current_target)
                self.score_label.setText(f"Streak: {self.score}")
            elif self.game_mode == "Contre la Montre":
                self.score_label.setText(f"Regions Found: {self.score}/{len(self.all_regions)}")
            QMessageBox.information(self, "Correct!", f"You found the {target_name}!")
            self.guess_button.setEnabled(False)
            self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
            self.consecutive_errors = 0
            for view in self.slice_views:
                view.stop_blinking()
            if self.game_mode == "Contre la Montre" and not self.remaining_regions:
                self.end_game()
            else:
                self.select_new_target()
        else:
            self.errors += 1
            self.consecutive_errors += 1
            self.incorrect_guesses.append((target_name, clicked_name))
            self.error_label.setText(f"Errors: {self.errors}")
            if self.game_mode == "Practice":
                self.score_label.setText(f"Correct: {self.score}")
                if self.consecutive_errors >= 3:
                    QMessageBox.warning(self, "Incorrect", f"That's the {clicked_name}.\nFind the {target_name}.\nThe correct region is now blinking!")
                    for view in self.slice_views:
                        view.start_blinking()
                else:
                    QMessageBox.warning(self, "Incorrect", f"That's the {clicked_name}.\nFind the {target_name}.")
            elif self.game_mode == "Streak":
                self.end_game()
            else:
                self.score_label.setText(f"Regions Found: {self.score}/{len(self.all_regions)}")
                QMessageBox.warning(self, "Incorrect", f"That's the {clicked_name}.\nFind the {target_name}.")
            self.guess_button.setEnabled(True)
            self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")

    def update_timer(self):
        if self.game_mode != "Contre la Montre":
            return
        self.total_time = int(time.time() - self.start_time)
        minutes = self.total_time // 60
        seconds = self.total_time % 60
        self.timer_label.setText(f"Time: {minutes}:{seconds:02d}")

    def end_game(self):
        self.game_running = False
        self.game_timer.stop()
        
        if self.game_mode in ["Practice", "Contre la Montre"]:
            if self.score > 0:
                accuracy = (self.score / (self.score + self.errors)) * 100
            else:
                accuracy = 0.0 if self.errors > 0 else 100.0
            current_pr = self.pr_data.get(self.current_atlas, {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0})
            
            if accuracy > current_pr["best_ratio"]:
                self.pr_data[self.current_atlas]["best_ratio"] = accuracy
                self.save_pr()
                self.update_pr_label()
                if accuracy == 100.0:
                    QMessageBox.information(self, "Perfect Run!", f"Perfect run with 100% accuracy for {self.current_atlas}!")
                else:
                    QMessageBox.information(self, "New Accuracy Record!", f"New best accuracy for {self.current_atlas}: {accuracy:.1f}%!")
        else:
            accuracy = 0.0
        
        if self.game_mode == "Contre la Montre":
            if self.errors == 0:
                current_time = self.total_time
                current_pr = self.pr_data.get(self.current_atlas, {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0})
                if current_time < current_pr["time"]:
                    self.pr_data[self.current_atlas]["time"] = current_time
                    self.save_pr()
                    self.update_pr_label()
                    QMessageBox.information(self, "New Personal Record!",
                                            f"New PR for {self.current_atlas}: {current_time // 60}'{current_time % 60:02d} \" !")
            recap = f"Game Over!\n\nAll regions found in {self.total_time} seconds.\n"
            recap += f"Accuracy: {accuracy:.1f}%\n"
            recap += f"Errors: {self.errors}\n"
            if self.incorrect_guesses:
                recap += "Incorrect guesses:\n" + "\n".join([f"- Looked for {target}, clicked {clicked}"
                                                           for target, clicked in self.incorrect_guesses])
            else:
                recap += "No errors."
        elif self.game_mode == "Streak":
            current_pr = self.pr_data.get(self.current_atlas, {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0})
            if self.score > current_pr["best_streak"]:
                self.pr_data[self.current_atlas]["best_streak"] = self.score
                self.save_pr()
                self.update_pr_label()
                QMessageBox.information(self, "New Streak Record!", f"New best streak for {self.current_atlas}: {self.score}!")
            recap = f"Game Over!\n\nStreak: {self.score}\n"
            if self.correct_guesses:
                recap += "Regions found:\n" + "\n".join([f"- {region}" for region in self.correct_guesses])
            else:
                recap += "No regions found."
        else:
            recap = f"Practice Ended!\n\nCorrect Guesses: {self.score}\n"
            recap += f"Accuracy: {accuracy:.1f}%\n"
            if self.correct_guesses:
                recap += "Regions found:\n" + "\n".join([f"- {region}" for region in self.correct_guesses]) + "\n\n"
            else:
                recap += "No regions found.\n\n"
            recap += f"Errors: {self.errors}\n"
            if self.incorrect_guesses:
                recap += "Incorrect guesses:\n" + "\n".join([f"- Looked for {target}, clicked {clicked}"
                                                           for target, clicked in self.incorrect_guesses])
            else:
                recap += "No errors."
        QMessageBox.information(self, "Game Over", recap)
        self.reset_game_ui()
        self.stacked_widget.setCurrentWidget(self.landing_widget)

    def show_help(self):
        if self.game_mode == "Practice":
            QMessageBox.information(self, "How to Play",
                                    "Practice:\n1. Select an atlas\n2. Choose 'Practice' mode\n"
                                    "3. Find regions with no time limit or score penalty\n4. Click or drag to move the crosshair\n"
                                    "5. Press Space or click 'Confirm Guess'\n6. After three errors on the same region, it will blink\n"
                                    "7. Toggle atlas visibility with 'Show Atlas Regions' checkbox\n"
                                    "8. View region information in the right panel\n9. Return to menu to end practice!")
        elif self.game_mode == "Contre la Montre":
            QMessageBox.information(self, "How to Play",
                                    "Contre la Montre:\n1. Select an atlas\n2. Choose 'Contre la Montre' mode\n"
                                    "3. Find all regions in the atlas as quickly as possible\n4. Click or drag to move the crosshair\n"
                                    "5. Press Space or click 'Confirm Guess'\n6. Toggle atlas visibility with 'Show Atlas Regions' checkbox\n"
                                    "7. Results show time taken and errors at the end!")
        else:
            QMessageBox.information(self, "How to Play",
                                    "Streak:\n1. Select an atlas\n2. Choose 'Streak' mode\n"
                                    "3. Find as many regions as possible without a mistake\n4. Click or drag to move the crosshair\n"
                                    "5. Press Space or click 'Confirm Guess'\n6. Toggle atlas visibility with 'Show Atlas Regions' checkbox\n"
                                    "7. Game ends on the first error!")

    def show_menu(self):
        self.game_running = False
        self.game_timer.stop()
        self.reset_game_ui()
        self.stacked_widget.setCurrentWidget(self.landing_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)
    game = NeuroGuessrGame()
    game.show()
    sys.exit(app.exec_())