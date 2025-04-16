import os
import sys
import time
import random
import numpy as np
import pandas as pd
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QGridLayout, QSplitter, QFrame,
                             QSlider, QProgressBar, QLCDNumber, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette, QImage, QKeySequence

class BrainSliceView(QLabel):
    """Widget to display a single brain slice with click, drag, and zoom functionality."""
    slice_clicked = pyqtSignal(int, int, int)  # x, y, plane_index

    def __init__(self, plane_index, parent=None):
        super().__init__(parent)
        self.plane_index = plane_index
        self.crosshair_pos = (0, 0)  # Position in 2D slice
        self.zoom_factor = 1.5
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.slice_data = None
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
        """Update crosshair position based on 3D coordinates."""
        if self.plane_index == 0:  # Axial (xy plane)
            self.crosshair_pos = (voxel_x, voxel_y)
        elif self.plane_index == 1:  # Coronal (xz plane)
            self.crosshair_pos = (voxel_x, voxel_z)
        elif self.plane_index == 2:  # Sagittal (yz plane)
            self.crosshair_pos = (voxel_y, voxel_z)
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if not self.original_pixmap:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.2  # Zoom in
        elif delta < 0:
            self.zoom_factor /= 1.2  # Zoom out
        self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
        self.update()
        event.accept()

    def update_slice(self, slice_data, colormap=None):
        """Update the displayed brain slice."""
        self.slice_data = slice_data
        if slice_data is None:
            self.clear()
            return
        if colormap is None:
            norm_slice = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
            h, w = norm_slice.shape
            qimg = QImage(norm_slice.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w = slice_data.shape
            colored_slice = np.zeros((h, w, 3), dtype=np.uint8)
            unique_vals = np.unique(slice_data)
            for val in unique_vals:
                if val in colormap:
                    mask = (slice_data == val)
                    colored_slice[mask] = colormap[val]
            qimg = QImage(colored_slice.data, w, h, w * 3, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        """Draw the slice with crosshairs and orientation labels."""
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
        
        # Draw crosshair
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        x, y = self.crosshair_pos
        scaled_x = int(x * self.zoom_factor) + x_offset
        scaled_y = int(y * self.zoom_factor) + y_offset
        painter.drawLine(scaled_x, y_offset, scaled_x, y_offset + img_height)
        painter.drawLine(x_offset, scaled_y, x_offset + img_width, scaled_y)
        
        # Draw orientation labels
        painter.setPen(QColor(255, 0, 0))
        font = QFont("Arial", 12)
        painter.setFont(font)
        if self.plane_index == 0:  # Axial
            painter.drawText(scaled_x - 30, y_offset - 10, "L")
            painter.drawText(scaled_x + 20, y_offset - 10, "R")
            painter.drawText(x_offset - 30, scaled_y - 10, "A")
            painter.drawText(x_offset - 30, scaled_y + 20, "P")
        elif self.plane_index == 1:  # Coronal
            painter.drawText(scaled_x - 30, y_offset - 10, "L")
            painter.drawText(scaled_x + 20, y_offset - 10, "R")
            painter.drawText(x_offset - 30, scaled_y - 10, "S")
            painter.drawText(x_offset - 30, scaled_y + 20, "I")
        elif self.plane_index == 2:  # Sagittal
            painter.drawText(scaled_x - 30, y_offset - 10, "A")
            painter.drawText(scaled_x + 20, y_offset - 10, "P")
            painter.drawText(x_offset - 30, scaled_y - 10, "S")
            painter.drawText(x_offset - 30, scaled_y + 20, "I")
        
        # Draw title
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(0, 20, label_width, 20, Qt.AlignCenter, self.plane_names[self.plane_index])
        painter.end()

    def mousePressEvent(self, event):
        """Set crosshair on click and start dragging."""
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
        """Update crosshair position smoothly while dragging."""
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
        """Stop dragging on mouse release."""
        if event.button() == Qt.LeftButton:
            self.dragging = False

class NeuroGuessrGame(QMainWindow):
    """Main window for the NeuroGuessr game."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGuessr - Test Your Brain Region Knowledge!")
        self.showMaximized()  # Start maximized
        self.score = 0
        self.errors = 0
        self.current_target = None
        self.game_running = False
        self.time_remaining = 60
        self.regions_to_find = 10
        self.regions_found = 0
        self.brain_data = None
        self.region_map = None
        self.colormap = {}
        self.current_slices = [None, None, None]
        self.current_positions = [0, 0, 0]
        self.selected_position = None
        self.crosshair_3d = (0, 0, 0)
        self.atlas_options = {
            "Brodmann" : ("/Users/francoisramon/Desktop/These/neuroguessr/data/brodmann_stride.nii.gz",
                          "/Users/francoisramon/Desktop/These/neuroguessr/data/brodmann.txt"),
            "APARC DKT": ("/Users/francoisramon/Desktop/These/neuroguessr/data/aparc_DKT_stride.nii.gz",
                          "/Users/francoisramon/Desktop/These/neuroguessr/data/fs_a2009s.txt"),
            "Harvard Oxford" : ("/Users/francoisramon/Desktop/These/neuroguessr/data/HarvardOxford_stride.nii.gz",
                          "/Users/francoisramon/Desktop/These/neuroguessr/data/HarvardOxford.txt"),
            "Thalamus": ("/Users/francoisramon/Desktop/These/neuroguessr/data/Thalamus_stride.nii.gz", "/Users/francoisramon/Desktop/These/neuroguessr/data/Thalamus.txt"),
            "Xtract": ("/Users/francoisramon/Desktop/These/neuroguessr/data/xtract_stride.nii.gz", "/Users/francoisramon/Desktop/These/neuroguessr/data/xtract.txt")
        }
        self.current_atlas = "Brodmann"
        self.setup_ui()
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_timer)
        self.load_data()

    def setup_ui(self):
        """Set up the user interface."""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Atlas selection
        atlas_layout = QHBoxLayout()
        atlas_label = QLabel("Select Atlas:")
        atlas_label.setStyleSheet("color: white;")
        self.atlas_combo = QComboBox()
        self.atlas_combo.addItems(self.atlas_options.keys())
        self.atlas_combo.setCurrentText(self.current_atlas)
        self.atlas_combo.currentIndexChanged.connect(self.change_atlas)
        atlas_layout.addWidget(atlas_label)
        atlas_layout.addWidget(self.atlas_combo)
        main_layout.addLayout(atlas_layout)
        
        # Game status area
        status_layout = QHBoxLayout()
        self.target_label = QLabel("Target: Not Started")
        self.target_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.target_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.timer_display = QLCDNumber()
        self.timer_display.setDigitCount(3)
        self.timer_display.display(60)
        self.timer_display.setStyleSheet("background-color: #333;")
        score_layout = QVBoxLayout()
        self.score_label = QLabel(f"Score: {self.score}")
        self.score_label.setFont(QFont("Arial", 14))
        self.error_label = QLabel(f"Errors: {self.errors}")
        self.error_label.setFont(QFont("Arial", 14))
        score_layout.addWidget(self.score_label)
        score_layout.addWidget(self.error_label)
        status_layout.addWidget(self.target_label, 3)
        status_layout.addWidget(self.timer_display, 1)
        status_layout.addLayout(score_layout, 1)
        main_layout.addLayout(status_layout)
        
        # Brain slice views
        views_layout = QHBoxLayout()
        self.slice_views = []
        for i in range(3):
            view = BrainSliceView(i)
            view.slice_clicked.connect(self.handle_slice_click)
            self.slice_views.append(view)
            views_layout.addWidget(view)
        main_layout.addLayout(views_layout, 1)
        
        # Sliders
        slider_layout = QHBoxLayout()
        z_layout = QVBoxLayout()
        z_label = QLabel("Axial (Z)")
        z_label.setAlignment(Qt.AlignCenter)
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(100)
        self.z_slider.setValue(50)
        self.z_slider.valueChanged.connect(lambda v: self.update_slice_position(0, v))
        z_layout.addWidget(z_label)
        z_layout.addWidget(self.z_slider)
        y_layout = QVBoxLayout()
        y_label = QLabel("Coronal (Y)")
        y_label.setAlignment(Qt.AlignCenter)
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.setMaximum(100)
        self.y_slider.setValue(50)
        self.y_slider.valueChanged.connect(lambda v: self.update_slice_position(1, v))
        y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_slider)
        x_layout = QVBoxLayout()
        x_label = QLabel("Sagittal (X)")
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
        main_layout.addLayout(slider_layout)
        
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
        main_layout.addLayout(button_layout)
        
        # Initially, hide guess and menu buttons
        self.guess_button.hide()
        self.menu_button.hide()
        
        # Bind space key to confirm guess
        self.setFocusPolicy(Qt.StrongFocus)
        self.keyPressEvent = self.handle_key_press
        
        self.setCentralWidget(main_widget)

    def handle_key_press(self, event):
        """Handle space key for confirming guess."""
        if event.key() == Qt.Key_Space and self.guess_button.isEnabled():
            self.validate_guess()

    def load_data(self):
        """Load brain data and region mapping."""
        atlas_name = self.current_atlas
        atlas_file, region_file = self.atlas_options[atlas_name]
        try:
            if os.path.exists(atlas_file) and os.path.exists(region_file):
                self.brain_data = nib.load(atlas_file)
                region_df = pd.read_csv(region_file, sep="\s+", comment="#", header=None, 
                                       names=["Index", "RegionName", "R", "G", "B", "A"])
                self.region_map = {row["Index"]: row["RegionName"] for _, row in region_df.iterrows()}
                self.colormap = {row["Index"]: (row["R"], row["G"], row["B"]) for _, row in region_df.iterrows()}
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
            QMessageBox.critical(self, "Error", f"Failed to load atlas {atlas_name}: {str(e)}\nUsing dummy data.")
            self.load_dummy_data()

    def load_dummy_data(self):
        """Load higher-resolution dummy data."""
        dummy_shape = (256, 256, 256)
        dummy_data = np.zeros(dummy_shape, dtype=np.int16)
        regions = {
            1: "Left Cerebral Cortex", 2: "Right Cerebral Cortex", 3: "Left Hippocampus",
            4: "Right Hippocampus", 5: "Left Thalamus", 6: "Right Thalamus",
            7: "Left Amygdala", 8: "Right Amygdala", 9: "Left Caudate", 10: "Right Caudate"
        }
        for i in range(1, 11):
            self.colormap[i] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            center = np.array([random.randint(30, 226), random.randint(30, 226), random.randint(30, 226)])
            size = random.randint(10, 20)
            if i % 2 == 0: center[0] = random.randint(130, 226)
            else: center[0] = random.randint(30, 126)
            for x in range(max(0, center[0]-size), min(dummy_shape[0], center[0]+size)):
                for y in range(max(0, center[1]-size), min(dummy_shape[1], center[1]+size)):
                    for z in range(max(0, center[2]-size), min(dummy_shape[2], center[2]+size)):
                        if ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2) < size**2:
                            dummy_data[x, y, z] = i
        affine = np.eye(4)
        self.brain_data = nib.Nifti1Image(dummy_data, affine)
        self.region_map = regions
        self.z_slider.setMaximum(dummy_shape[2] - 1)
        self.y_slider.setMaximum(dummy_shape[1] - 1)
        self.x_slider.setMaximum(dummy_shape[0] - 1)
        self.z_slider.setValue(dummy_shape[2] // 2)
        self.y_slider.setValue(dummy_shape[1] // 2)
        self.x_slider.setValue(dummy_shape[0] // 2)
        self.update_all_slices()

    def start_game(self):
        """Start or restart the game."""
        self.score = 0
        self.errors = 0
        self.regions_found = 0
        self.time_remaining = 60
        self.game_running = True
        self.score_label.setText(f"Score: {self.score}")
        self.error_label.setText(f"Errors: {self.errors}")
        self.timer_display.display(self.time_remaining)
        self.start_button.hide()
        self.guess_button.show()
        self.menu_button.show()
        self.guess_button.setEnabled(False)
        self.select_new_target()
        self.game_timer.start(1000)

    def change_atlas(self, index):
        """Change the atlas and reload data."""
        self.current_atlas = self.atlas_combo.currentText()
        self.load_data()

    def update_slice_position(self, plane_index, value):
        """Update slice position from slider."""
        self.current_positions[plane_index] = value
        if plane_index == 0: self.crosshair_3d = (self.crosshair_3d[0], self.crosshair_3d[1], value)
        elif plane_index == 1: self.crosshair_3d = (self.crosshair_3d[0], value, self.crosshair_3d[2])
        elif plane_index == 2: self.crosshair_3d = (value, self.crosshair_3d[1], self.crosshair_3d[2])
        self.update_all_slices()

    def update_all_slices(self):
        """Update all slice views."""
        if self.brain_data is None:
            return
        brain_3d = self.brain_data.get_fdata().astype(np.int32)
        z, y, x = self.current_positions
        brain_shape = self.brain_data.shape
        x = min(max(x, 0), brain_shape[0] - 1)
        y = min(max(y, 0), brain_shape[1] - 1)
        z = min(max(z, 0), brain_shape[2] - 1)
        axial_slice = brain_3d[:, :, z].T
        coronal_slice = brain_3d[:, y, :].T
        sagittal_slice = brain_3d[x, :, :].T
        self.slice_views[0].update_slice(axial_slice, self.colormap)
        self.slice_views[1].update_slice(coronal_slice, self.colormap)
        self.slice_views[2].update_slice(sagittal_slice, self.colormap)
        voxel_x, voxel_y, voxel_z = self.crosshair_3d
        for view in self.slice_views:
            view.set_crosshair_3d(voxel_x, voxel_y, voxel_z)

    def select_new_target(self):
        """Select a new target region."""
        valid_regions = [int(val) for val in np.unique(self.brain_data.get_fdata().astype(np.int32)) 
                        if val > 0 and val in self.region_map]
        if not valid_regions:
            QMessageBox.warning(self, "Error", "No valid regions found.")
            return
        region_id = random.choice(valid_regions)
        self.current_target = region_id
        self.target_label.setText(f"Find: {self.region_map[region_id]}")

    def handle_slice_click(self, x, y, plane_index):
        """Handle slice click or drag."""
        if not self.game_running or not self.brain_data:
            return
        brain_shape = self.brain_data.shape
        if plane_index == 0:  # Axial
            voxel_x = min(max(x, 0), brain_shape[0] - 1)
            voxel_y = min(max(y, 0), brain_shape[1] - 1)
            voxel_z = self.current_positions[0]
        elif plane_index == 1:  # Coronal
            voxel_x = min(max(x, 0), brain_shape[0] - 1)
            voxel_y = self.current_positions[1]
            voxel_z = min(max(y, 0), brain_shape[2] - 1)
        else:  # Sagittal
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
        """Validate the player's guess."""
        if not self.selected_position or not self.game_running:
            return
        voxel_x, voxel_y, voxel_z = self.selected_position
        brain_data = self.brain_data.get_fdata().astype(np.int32)
        try:
            clicked_region = int(brain_data[voxel_x, voxel_y, voxel_z])
        except IndexError:
            clicked_region = 0
        if clicked_region == self.current_target:
            self.score += 1
            self.regions_found += 1
            self.score_label.setText(f"Score: {self.score}")
            region_name = self.region_map.get(clicked_region, "Unknown")
            QMessageBox.information(self, "Correct!", f"You found the {region_name}!")
            if self.regions_found >= self.regions_to_find:
                self.end_game()
            else:
                self.time_remaining = 60
                self.timer_display.display(self.time_remaining)
                self.select_new_target()
        else:
            self.errors += 1
            self.error_label.setText(f"Errors: {self.errors}")
            clicked_name = self.region_map.get(clicked_region, "Background/Unknown")
            target_name = self.region_map.get(self.current_target, "Unknown")
            QMessageBox.warning(self, "Incorrect", f"That's the {clicked_name}.\nFind the {target_name}.")
        self.guess_button.setEnabled(False)
        self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")

    def update_timer(self):
        """Update the game timer."""
        self.time_remaining -= 1
        self.timer_display.display(self.time_remaining)
        if self.time_remaining <= 0:
            self.errors += 1
            self.error_label.setText(f"Errors: {self.errors}")
            QMessageBox.warning(self, "Time's Up!", "Time ran out!")
            self.regions_found += 1
            if self.regions_found >= self.regions_to_find:
                self.end_game()
            else:
                self.time_remaining = 60
                self.timer_display.display(self.time_remaining)
                self.select_new_target()

    def end_game(self):
        """End the game."""
        self.game_running = False
        self.game_timer.stop()
        QMessageBox.information(self, "Game Over", 
                              f"Final Score: {self.score}/{self.regions_to_find}\nErrors: {self.errors}")
        self.start_button.setText("Start New Game")
        self.target_label.setText("Game Over")
        self.show_menu()

    def show_help(self):
        """Show help dialog."""
        QMessageBox.information(self, "How to Play", 
                             "1. Select an atlas\n2. Click 'Start Game'\n3. Find the target region\n"
                             "4. Click or drag to move the crosshair\n5. Press Space or click 'Confirm Guess'\n"
                             "6. Score points for correct guesses!")

    def show_menu(self):
        """Show the start menu."""
        self.game_running = False
        self.game_timer.stop()
        self.start_button.show()
        self.guess_button.hide()
        self.menu_button.hide()
        self.target_label.setText("Target: Not Started")
        self.timer_display.display(60)
        self.score_label.setText("Score: 0")
        self.error_label.setText("Errors: 0")

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