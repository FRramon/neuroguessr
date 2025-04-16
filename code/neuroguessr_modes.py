import os
import sys
import random
import numpy as np
import pandas as pd
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QMessageBox, QComboBox,
                             QStackedWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette, QImage, QKeySequence

class BrainSliceView(QLabel):
    """Widget to display a single brain slice with click, drag, and zoom functionality."""
    slice_clicked = pyqtSignal(int, int, int)  # x, y, plane_index

    def __init__(self, plane_index, parent=None):
        super().__init__(parent)
        self.plane_index = plane_index
        self.crosshair_pos = (0, 0)
        self.zoom_factor = 1.5
        self.setMinimumSize(400, 400)
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
        
        self.title = QLabel(self.plane_names[plane_index])
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 0);")
        self.title.setFont(QFont("Arial", 12, QFont.Bold))

    def set_crosshair_3d(self, voxel_x, voxel_y, voxel_z):
        if self.plane_index == 0:
            self.crosshair_pos = (voxel_x, voxel_y)
        elif self.plane_index == 1:
            self.crosshair_pos = (voxel_x, voxel_z)
        elif self.plane_index == 2:
            self.crosshair_pos = (voxel_y, voxel_z)
        self.update()

    def wheelEvent(self, event):
        if not self.original_pixmap:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.2
        elif delta < 0:
            self.zoom_factor /= 1.2
        self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
        self.update()
        event.accept()

    def update_slice(self, slice_data, template_slice, colormap=None):
        self.slice_data = slice_data
        self.template_data = template_slice
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
        if colormap:
            unique_vals = np.unique(slice_data)
            for val in unique_vals:
                if val in colormap and val > 0:
                    mask = (slice_data == val)
                    color = colormap[val]
                    colored_slice[mask, 0] = (0.5 * colored_slice[mask, 0] + 0.5 * color[0]).astype(np.uint8)
                    colored_slice[mask, 1] = (0.5 * colored_slice[mask, 1] + 0.5 * color[1]).astype(np.uint8)
                    colored_slice[mask, 2] = (0.5 * colored_slice[mask, 2] + 0.5 * color[2]).astype(np.uint8)
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
        font = QFont("Arial", 12)
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
        painter.setFont(QFont("Arial", 12, QFont.Bold))
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
    """Main window for the NeuroGuessr game with landing page and two modes."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGuessr")
        self.showMaximized()
        self.score = 0
        self.errors = 0
        self.current_target = None
        self.game_running = False
        self.game_mode = "Contre la Montre"
        self.time_remaining = 120
        self.correct_guesses = []
        self.incorrect_guesses = []
        self.streak_guessed_regions = []
        self.brain_data = None
        self.template_data = None
        self.region_map = None
        self.colormap = {}
        self.current_slices = [None, None, None]
        self.current_positions = [0, 0, 0]
        self.selected_position = None
        self.crosshair_3d = (0, 0, 0)
        self.atlas_options = {
            "Brodmann": ("/Users/francoisramon/Desktop/These/neuroguessr/data/brodmann_grid_stride.nii.gz",
                         "/Users/francoisramon/Desktop/These/neuroguessr/data/brodmann.txt"),
            "Harvard Oxford": ("/Users/francoisramon/Desktop/These/neuroguessr/data/HarvardOxford-cort-maxprob-thr25-1mm_stride.nii.gz",
                              "/Users/francoisramon/Desktop/These/neuroguessr/data/HarvardOxford-Cortical.txt"),
            "Subcortical": ("/Users/francoisramon/Desktop/These/neuroguessr/data/HarvardOxford-sub-maxprob-thr25-1mm_stride.nii.gz",
                           "/Users/francoisramon/Desktop/These/neuroguessr/data/HarvardOxford-Subcortical.txt"),
            "Thalamus": ("/Users/francoisramon/Desktop/These/neuroguessr/data/Thalamus_stride.nii.gz",
                         "/Users/francoisramon/Desktop/These/neuroguessr/data/Thalamus.txt"),
            "Cerebellum": ("/Users/francoisramon/Desktop/These/neuroguessr/data/Cerebellum-MNIfnirt-maxprob-thr25-1mm_stride.nii.gz",
                          "/Users/francoisramon/Desktop/These/neuroguessr/data/Cerebellum_MNIfnirt.txt"),
            "Xtract": ("/Users/francoisramon/Desktop/These/neuroguessr/data/xtract_stride.nii.gz",
                       "/Users/francoisramon/Desktop/These/neuroguessr/data/xtract.txt")
        }
        self.current_atlas = "Brodmann"
        self.setup_ui()
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_timer)

    def setup_ui(self):
        """Set up the main UI with a stacked widget for landing page and game."""
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Landing page
        self.landing_widget = QWidget()
        landing_layout = QVBoxLayout(self.landing_widget)
        landing_layout.setAlignment(Qt.AlignCenter)
        landing_layout.setSpacing(20)

        # Logo (text-based placeholder)
        logo_label = QLabel("NeuroGuessr\nBrain Atlas Challenge")
        logo_label.setFont(QFont("Arial", 36, QFont.Bold))
        logo_label.setStyleSheet("color: white;")
        logo_label.setAlignment(Qt.AlignCenter)
        landing_layout.addWidget(logo_label)

        # Game mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Game Mode:")
        mode_label.setStyleSheet("color: white; font-size: 16px;")
        self.landing_mode_combo = QComboBox()
        self.landing_mode_combo.addItems(["Contre la Montre", "Streak"])
        self.landing_mode_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.landing_mode_combo)
        landing_layout.addLayout(mode_layout)

        # Atlas selection
        atlas_layout = QHBoxLayout()
        atlas_label = QLabel("Atlas:")
        atlas_label.setStyleSheet("color: white; font-size: 16px;")
        self.landing_atlas_combo = QComboBox()
        self.landing_atlas_combo.addItems(self.atlas_options.keys())
        self.landing_atlas_combo.setCurrentText(self.current_atlas)
        self.landing_atlas_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        atlas_layout.addWidget(atlas_label)
        atlas_layout.addWidget(self.landing_atlas_combo)
        landing_layout.addLayout(atlas_layout)

        # Play button
        play_button = QPushButton("Play")
        play_button.setStyleSheet("font-size: 18px; padding: 10px; background-color: #4CAF50; color: white;")
        play_button.clicked.connect(self.start_game_from_landing)
        landing_layout.addWidget(play_button)

        # Quit button
        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet("font-size: 18px; padding: 10px; background-color: #f44336; color: white;")
        quit_button.clicked.connect(QApplication.instance().quit)
        landing_layout.addWidget(quit_button)

        landing_layout.addStretch()
        self.stacked_widget.addWidget(self.landing_widget)

        # Game UI
        self.game_widget = QWidget()
        game_layout = QVBoxLayout(self.game_widget)

        # Atlas selection (in-game)
        selection_layout = QHBoxLayout()
        atlas_layout = QHBoxLayout()
        atlas_label = QLabel("Select Atlas:")
        atlas_label.setStyleSheet("color: white;")
        self.game_atlas_combo = QComboBox()
        self.game_atlas_combo.addItems(self.atlas_options.keys())
        self.game_atlas_combo.setCurrentText(self.current_atlas)
        self.game_atlas_combo.currentIndexChanged.connect(self.change_atlas)
        atlas_layout.addWidget(atlas_label)
        atlas_layout.addWidget(self.game_atlas_combo)
        selection_layout.addLayout(atlas_layout)
        game_layout.addLayout(selection_layout)

        # Game status area
        status_layout = QHBoxLayout()
        self.target_label = QLabel("Target: Not Started")
        self.target_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.target_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.timer_label = QLabel("Time: 2:00")
        self.timer_label.setFont(QFont("Arial", 14))
        self.timer_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        score_layout = QVBoxLayout()
        self.score_label = QLabel("Correct: 0")
        self.score_label.setFont(QFont("Arial", 14))
        self.error_label = QLabel("Errors: 0")
        self.error_label.setFont(QFont("Arial", 14))
        score_layout.addWidget(self.score_label)
        score_layout.addWidget(self.error_label)
        status_layout.addWidget(self.target_label, 3)
        status_layout.addWidget(self.timer_label, 1)
        status_layout.addLayout(score_layout, 1)
        game_layout.addLayout(status_layout)

        # Brain slice views
        views_layout = QHBoxLayout()
        self.slice_views = []
        for i in range(3):
            view = BrainSliceView(i)
            view.slice_clicked.connect(self.handle_slice_click)
            self.slice_views.append(view)
            views_layout.addWidget(view)
        game_layout.addLayout(views_layout, 1)

        # Sliders
        slider_layout = QHBoxLayout()
        z_layout = QVBoxLayout()
        z_label = QLabel("Axial")
        z_label.setAlignment(Qt.AlignCenter)
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

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Game")
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
        self.guess_button = QPushButton("Confirm Guess")
        self.guess_button.clicked.connect(self.validate_guess)
        self.guess_button.setEnabled(False)
        self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        self.menu_button = QPushButton("Menu")
        self.menu_button.clicked.connect(self.show_menu)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(QApplication.instance().quit)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.guess_button)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.menu_button)
        button_layout.addWidget(self.quit_button)
        game_layout.addLayout(button_layout)

        self.guess_button.hide()
        self.menu_button.hide()

        self.setFocusPolicy(Qt.StrongFocus)
        self.keyPressEvent = self.handle_key_press

        self.stacked_widget.addWidget(self.game_widget)

        # Show landing page initially
        self.stacked_widget.setCurrentWidget(self.landing_widget)
        self.load_data()

    def start_game_from_landing(self):
        """Transition from landing page to game UI."""
        self.game_mode = self.landing_mode_combo.currentText()
        self.current_atlas = self.landing_atlas_combo.currentText()
        self.game_atlas_combo.setCurrentText(self.current_atlas)
        self.load_data()
        self.set_game_mode(self.game_mode)
        self.stacked_widget.setCurrentWidget(self.game_widget)

    def handle_key_press(self, event):
        if event.key() == Qt.Key_Space and self.guess_button.isEnabled():
            self.validate_guess()

    def load_data(self):
        atlas_name = self.current_atlas
        atlas_file, region_file = self.atlas_options[atlas_name]
        template_file = "/Users/francoisramon/Desktop/These/neuroguessr/data/MNI_template_1mm_stride.nii.gz"
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
        if mode == "Streak":
            self.timer_label.setText("Time: N/A")
            self.score_label.setText("Streak: 0")
        else:
            self.timer_label.setText("Time: 2:00")
            self.score_label.setText("Correct: 0")

    def start_game(self):
        self.score = 0
        self.errors = 0
        self.correct_guesses = []
        self.incorrect_guesses = []
        self.streak_guessed_regions = []
        self.game_running = True
        self.score_label.setText(f"{'Streak' if self.game_mode == 'Streak' else 'Correct'}: {self.score}")
        self.error_label.setText("Errors: 0")
        if self.game_mode == "Contre la Montre":
            self.time_remaining = 120
            self.update_timer_display()
        else:
            self.timer_label.setText("Time: N/A")
        self.start_button.hide()
        self.guess_button.show()
        self.menu_button.show()
        self.guess_button.setEnabled(False)
        self.select_new_target()
        if self.game_mode == "Contre la Montre":
            self.game_timer.start(1000)

    def update_timer_display(self):
        minutes = self.time_remaining // 60
        seconds = self.time_remaining % 60
        self.timer_label.setText(f"Time: {minutes}:{seconds:02d}")

    def change_atlas(self):
        self.current_atlas = self.game_atlas_combo.currentText()
        self.load_data()

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
        self.slice_views[0].update_slice(axial_slice, axial_template, self.colormap)
        self.slice_views[1].update_slice(coronal_slice, coronal_template, self.colormap)
        self.slice_views[2].update_slice(sagittal_slice, sagittal_template, self.colormap)
        voxel_x, voxel_y, voxel_z = self.crosshair_3d
        for view in self.slice_views:
            view.set_crosshair_3d(voxel_x, voxel_y, voxel_z)

    def select_new_target(self):
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
        self.target_label.setText(f"Find: {self.region_map[region_id]}")

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
            if self.game_mode == "Streak":
                self.streak_guessed_regions.append(self.current_target)
                self.score_label.setText(f"Streak: {self.score}")
            else:
                self.score_label.setText(f"Correct: {self.score}")
            QMessageBox.information(self, "Correct!", f"You found the {target_name}!")
            self.guess_button.setEnabled(False)
            self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
            self.select_new_target()
        else:
            self.errors += 1
            self.incorrect_guesses.append((target_name, clicked_name))
            self.error_label.setText(f"Errors: {self.errors}")
            QMessageBox.warning(self, "Incorrect", f"That's the {clicked_name}.\nFind the {target_name}.")
            if self.game_mode == "Streak":
                self.end_game()
            else:
                self.guess_button.setEnabled(False)
                self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
                self.select_new_target()

    def update_timer(self):
        if self.game_mode != "Contre la Montre":
            return
        self.time_remaining -= 1
        self.update_timer_display()
        if self.time_remaining <= 0:
            self.end_game()

    def end_game(self):
        self.game_running = False
        self.game_timer.stop()
        if self.game_mode == "Contre la Montre":
            recap = f"Time's Up!\n\nCorrect Guesses: {self.score}\n"
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
        else:
            recap = f"Game Over!\n\nStreak: {self.score}\n"
            if self.correct_guesses:
                recap += "Regions found:\n" + "\n".join([f"- {region}" for region in self.correct_guesses])
            else:
                recap += "No regions found."
        QMessageBox.information(self, "Game Over", recap)
        self.stacked_widget.setCurrentWidget(self.landing_widget)

    def show_help(self):
        if self.game_mode == "Contre la Montre":
            QMessageBox.information(self, "How to Play",
                                    "Contre la Montre:\n1. Select an atlas\n2. Choose 'Contre la Montre' mode\n"
                                    "3. Find as many regions as possible in 2 minutes\n4. Click or drag to move the crosshair\n"
                                    "5. Press Space or click 'Confirm Guess'\n6. Results are shown at the end!")
        else:
            QMessageBox.information(self, "How to Play",
                                    "Streak:\n1. Select an atlas\n2. Choose 'Streak' mode\n"
                                    "3. Find as many regions as possible without a mistake\n4. Click or drag to move the crosshair\n"
                                    "5. Press Space or click 'Confirm Guess'\n6. Game ends on the first error!")

    def show_menu(self):
        self.game_running = False
        self.game_timer.stop()
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