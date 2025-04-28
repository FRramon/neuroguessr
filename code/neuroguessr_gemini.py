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
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette, QImage, QFontDatabase, QIcon

def get_resource_path(relative_path):
    """Get the absolute path to a resource, works for both development and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        # Normal development path
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
        # Initial background set by NeuroGuessrGame after creation
        self.setStyleSheet("background-color: black;")
        self.slice_data = None
        self.template_data = None
        self.plane_names = ["Axial", "Coronal", "Sagittal"]
        self.original_pixmap = None
        self.setMouseTracking(True) # Enable mouse tracking even when no button is pressed
        self.setFocusPolicy(Qt.WheelFocus) # Focus policy to receive wheel events
        self.dragging = False
        self.last_mouse_pos = None
        self.blinking = False
        self.blink_state = True
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink)

        # Title Label (relative to this widget)
        self.title = QLabel(self.plane_names[plane_index], self)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 0);") # Transparent background
        self.title.setFont(QFont("Helvetica [Cronyx]", 12, QFont.Bold))
        # Position the title - adjust as needed
        self.title.setGeometry(0, 5, self.width(), 20) # x, y, width, height

    def set_background(self, color_name):
        """Sets the background color of the widget."""
        self.setStyleSheet(f"background-color: {color_name};")
        # Adjust text color based on background for visibility
        text_color = "white" if color_name == "black" else "black"
        self.title.setStyleSheet(f"color: {text_color}; background-color: rgba(0, 0, 0, 0);")


    def start_blinking(self):
        self.blinking = True
        self.blink_timer.start(500) # Blink interval 500ms

    def stop_blinking(self):
        self.blinking = False
        self.blink_timer.stop()
        self.blink_state = True # Ensure it stops in the visible state
        # Force a redraw in the non-blinking state
        self.update_slice(self.slice_data, self.template_data, self.colormap, self.highlight_region, self.show_atlas)

    def toggle_blink(self):
        self.blink_state = not self.blink_state
        # Force redraw to show/hide blinking region
        self.update_slice(self.slice_data, self.template_data, self.colormap, self.highlight_region, self.show_atlas)

    def set_crosshair_3d(self, voxel_x, voxel_y, voxel_z):
        """Sets the crosshair position based on 3D voxel coordinates."""
        # Map 3D coordinates to 2D plane coordinates
        if self.plane_index == 0: # Axial
            self.crosshair_pos = (voxel_x, voxel_y)
        elif self.plane_index == 1: # Coronal
            self.crosshair_pos = (voxel_x, voxel_z)
        elif self.plane_index == 2: # Sagittal
            self.crosshair_pos = (voxel_y, voxel_z)
        self.update() # Trigger repaint

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming (Ctrl+Wheel) or slicing (Wheel)."""
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier: # Check if Ctrl (or Cmd on Mac) is pressed
            # Zooming
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.1
            elif delta < 0:
                self.zoom_factor /= 1.1
            # Clamp zoom factor
            self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
            self.update() # Trigger repaint for zoom
        else:
            # Slicing
            delta = event.angleDelta().y()
            if delta != 0:
                step = 1 if delta > 0 else -1
                self.slice_changed.emit(self.plane_index, step) # Emit signal to change slice
        event.accept()

    def update_slice(self, slice_data, template_slice, colormap=None, highlight_region=None, show_atlas=True):
        """Updates the displayed slice with optional atlas overlay and highlighting."""
        self.slice_data = slice_data
        self.template_data = template_slice
        self.colormap = colormap
        self.highlight_region = highlight_region
        self.show_atlas = show_atlas

        if slice_data is None or template_slice is None:
            self.clear()
            self.original_pixmap = None
            return

        # Normalize template slice for grayscale background
        norm_template = ((template_slice - template_slice.min()) /
                        (template_slice.max() - template_slice.min() + 1e-8) * 255).astype(np.uint8)

        h, w = norm_template.shape
        # Create RGB image: initialize with grayscale template
        colored_slice = np.zeros((h, w, 3), dtype=np.uint8)
        colored_slice[:, :, 0] = norm_template
        colored_slice[:, :, 1] = norm_template
        colored_slice[:, :, 2] = norm_template

        # Overlay colored atlas if enabled and colormap exists
        if show_atlas and colormap:
            unique_vals = np.unique(slice_data)
            for val in unique_vals:
                if val in colormap and val > 0: # Exclude background (0)
                    mask = (slice_data == val)
                    color = colormap[val]
                    # Handle blinking for the highlight region
                    if self.blinking and val == highlight_region and self.blink_state:
                        # Make blinking region bright yellow
                        colored_slice[mask, 0] = 255
                        colored_slice[mask, 1] = 255
                        colored_slice[mask, 2] = 0
                    else:
                        # Blend atlas color with template background
                        colored_slice[mask, 0] = (0.5 * colored_slice[mask, 0] + 0.5 * color[0]).astype(np.uint8)
                        colored_slice[mask, 1] = (0.5 * colored_slice[mask, 1] + 0.5 * color[1]).astype(np.uint8)
                        colored_slice[mask, 2] = (0.5 * colored_slice[mask, 2] + 0.5 * color[2]).astype(np.uint8)
        # Handle blinking even if atlas overlay is off
        elif self.blinking and highlight_region is not None and self.blink_state:
             mask = (slice_data == highlight_region)
             # Make blinking region bright yellow on the template background
             colored_slice[mask, 0] = 255
             colored_slice[mask, 1] = 255
             colored_slice[mask, 2] = 0


        # Convert numpy array to QImage/QPixmap
        qimg = QImage(colored_slice.data, w, h, w * 3, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update() # Trigger repaint

    def paintEvent(self, event):
        """Overrides paint event to draw the zoomed/panned slice and crosshair."""
        super().paintEvent(event) # Draw the base QLabel content (background)
        if not self.original_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)


        # Calculate scaled pixmap dimensions and position for centering
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        label_width = self.width()
        label_height = self.height()

        # Center the image
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2

        # Draw the scaled pixmap
        scaled_pixmap = self.original_pixmap.scaled(img_width, img_height,
                                                  Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)

        # Draw crosshair
        pen = QPen(QColor(255, 0, 0)) # Red crosshair
        pen.setWidth(1)
        painter.setPen(pen)

        # Convert crosshair position from original image coords to widget coords
        orig_x, orig_y = self.crosshair_pos
        scaled_x = int(orig_x * self.zoom_factor) + x_offset
        scaled_y = int(orig_y * self.zoom_factor) + y_offset

        # Draw crosshair lines spanning the visible pixmap area
        painter.drawLine(scaled_x, y_offset, scaled_x, y_offset + img_height) # Vertical line
        painter.drawLine(x_offset, scaled_y, x_offset + img_width, scaled_y) # Horizontal line

        # Draw orientation labels (relative to the pixmap edges) - Adjust offsets as needed
        text_pen_color = QColor(255, 255, 255) if self.styleSheet().split(':')[1].strip().lower().startswith('black') else QColor(0, 0, 0)
        painter.setPen(text_pen_color)
        font = QFont("Helvetica [Cronyx]", 10, QFont.Bold)
        painter.setFont(font)
        label_offset = 5 # Pixels away from the border

        # Top-Bottom labels
        painter.drawText(x_offset + img_width // 2 - 5, y_offset - label_offset, "A" if self.plane_index != 2 else "S") # Anterior/Superior
        painter.drawText(x_offset + img_width // 2 - 5, y_offset + img_height + label_offset + 10, "P" if self.plane_index != 2 else "I") # Posterior/Inferior

        # Left-Right labels
        painter.drawText(x_offset - label_offset - 10, y_offset + img_height // 2 + 5, "L" if self.plane_index != 0 else "S") # Left/Superior (or R?) Check convention
        painter.drawText(x_offset + img_width + label_offset, y_offset + img_height // 2 + 5, "R" if self.plane_index != 0 else "I") # Right/Inferior (or L?) Check convention

        # Redraw the title (ensure it's on top) - Already done by creating it as a child
        # self.title.setGeometry(0, 5, self.width(), 20) # Update geometry in case of resize
        # self.title.raise_() # Ensure title is drawn on top

        painter.end()


    def mousePressEvent(self, event):
        """Handle mouse press events for clicking or starting a drag."""
        if not self.original_pixmap or event.button() != Qt.LeftButton:
            return

        # Calculate image position based on click
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        label_width = self.width()
        label_height = self.height()
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2

        # Convert click coordinates (widget) to original image coordinates
        x = event.x() - x_offset
        y = event.y() - y_offset
        orig_x = int(x / self.zoom_factor)
        orig_y = int(y / self.zoom_factor)

        # Check if click is within the image bounds
        if 0 <= orig_x < self.original_pixmap.width() and 0 <= orig_y < self.original_pixmap.height():
            self.crosshair_pos = (orig_x, orig_y)
            self.update() # Trigger repaint to show new crosshair position
            # Emit signal with original image coordinates and plane index
            self.slice_clicked.emit(orig_x, orig_y, self.plane_index)

        # For dragging functionality (optional, can be complex with zooming)
        self.dragging = True
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging the crosshair."""
        # Only update crosshair if dragging and left button is pressed
        if not self.original_pixmap or not self.dragging or not (event.buttons() & Qt.LeftButton):
             return

        # Calculate image position based on mouse move
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        label_width = self.width()
        label_height = self.height()
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2

        # Convert mouse coordinates (widget) to original image coordinates
        x = event.x() - x_offset
        y = event.y() - y_offset
        orig_x = int(x / self.zoom_factor)
        orig_y = int(y / self.zoom_factor)

        # Clamp coordinates to be within the image bounds
        orig_x = max(0, min(orig_x, self.original_pixmap.width() - 1))
        orig_y = max(0, min(orig_y, self.original_pixmap.height() - 1))

        # Update crosshair position only if it has changed
        if self.crosshair_pos != (orig_x, orig_y):
            self.crosshair_pos = (orig_x, orig_y)
            self.update() # Trigger repaint
            # Emit signal with updated original image coordinates
            self.slice_clicked.emit(orig_x, orig_y, self.plane_index)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop dragging."""
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def resizeEvent(self, event):
        """Handle resize event to reposition the title."""
        super().resizeEvent(event)
        # Reposition title when the widget is resized
        self.title.setGeometry(0, 5, self.width(), 20)


class NeuroGuessrGame(QMainWindow):
    """Main window for the NeuroGuessr game with landing page and three modes."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGuessr")
        self.showMaximized() # Start maximized

        # --- Game State Variables ---
        self.score = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.current_target = None # ID of the target region
        self.game_running = False
        self.game_mode = "Practice" # "Practice", "Contre la Montre", "Streak"
        self.time_remaining = 180 # For Contre la Montre (seconds)
        self.correct_guesses = [] # List of correctly guessed region names
        self.incorrect_guesses = [] # List of tuples (target_name, clicked_name)
        self.streak_guessed_regions = [] # List of region IDs guessed correctly in streak mode
        self.start_time = None # For Contre la Montre timer
        self.total_time = 0 # For Contre la Montre timer

        # --- Data Variables ---
        self.brain_data = None # Nifti image object for atlas
        self.template_data = None # Nifti image object for background template
        self.region_map = None # Dictionary mapping region ID to region name
        self.colormap = {} # Dictionary mapping region ID to RGB tuple
        self.region_info = {} # Dictionary mapping region ID (str) to detailed info (structure, function)
        self.all_regions = [] # List of all valid region IDs in the current atlas
        self.remaining_regions = [] # List of regions yet to be found in Contre la Montre

        # --- UI State Variables ---
        self.current_slices = [None, None, None] # Actual slice data for each view (not really needed with direct access)
        self.current_positions = [0, 0, 0] # Current slice indices [z, y, x]
        self.selected_position = None # Last clicked voxel coordinates (x, y, z)
        self.crosshair_3d = (0, 0, 0) # Shared 3D crosshair position (voxel_x, voxel_y, voxel_z)
        self.show_atlas = True # Toggle visibility of atlas overlay
        self.use_colored_atlas = True # Landing page setting: use colors or not
        self.current_atlas = "AAL" # Default atlas
        self.current_template_name = "MNI" # Default template ("MNI" or "BigBrain")

        # --- File Paths & Configuration ---
        self.pr_file = os.path.join(Path.home(), ".neuroguessr", "pr.json") # Personal records file
        # Atlas definitions: (nifti_path, labels_path)
        self.atlas_options = {
            "AAL": (
                get_resource_path("data/aal_registered_nearest.nii.gz"),
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
                get_resource_path("data/ICBM2009b_asym-SubCorSeg-1mm_nn_stride.nii.gz"),
                get_resource_path("data/subcortical_bb.txt")
            ),
            "Cerebellum": (
                get_resource_path("data/Cerebellum-MNIfnirt-maxprob-thr25-1mm_stride.nii.gz"),
                get_resource_path("data/Cerebellum_MNIfnirt.txt")
            ),
            "Xtract": (
                get_resource_path("data/xtract_stride.nii.gz"),
                get_resource_path("data/xtract.txt")
            ) ,
            "Thalamus": (
                get_resource_path("data/Thalamus-thr0_stride_nn_sub.nii.gz"),
                get_resource_path("data/thalamus_lut.txt")
            )  ,
            "Brain Stem": (
                get_resource_path("data/Brainstem-thr0_stride_nn_sub.nii.gz"),
                get_resource_path("data/brainstem_lut.txt")
            )  ,
            "Hippocampus Amygdala": (
                get_resource_path("data/HippoAmyg_left-thr0_stride_nn_sub.nii.gz"),
                get_resource_path("data/hippoamyg_left_lut.txt")
            )
        }
        # Template file paths
        self.template_files = {
            "MNI": get_resource_path("data/MNI_template_1mm_stride.nii.gz"),
            "BigBrain": get_resource_path("data/BigBrain_ICBM_regrid_stride.nii.gz") # ADJUST PATH IF NEEDED
        }

        # --- Load Personal Records ---
        self.pr_data = self.load_pr()

        # --- Setup UI ---
        self.setup_ui()

        # --- Timers ---
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_timer) # Timer for Contre la Montre

    def load_pr(self):
        """Load personal records from JSON file, migrating old data to colored mode."""
        os.makedirs(os.path.dirname(self.pr_file), exist_ok=True)
        default_pr_structure = lambda: {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0}
        try:
            with open(self.pr_file, 'r') as f:
                data = json.load(f)
                # Ensure all known atlases have entries and the correct structure
                for atlas in self.atlas_options.keys():
                    if atlas not in data:
                        # Atlas not present, create default structure
                        data[atlas] = {
                            "colored": default_pr_structure(),
                            "non_colored": default_pr_structure()
                        }
                    else:
                        # Atlas exists, check structure
                        if "time" in data[atlas]: # Old format detected
                            old_pr = data[atlas]
                            data[atlas] = {
                                "colored": { # Assume old data was for colored mode
                                    "time": old_pr.get("time", float("inf")),
                                    "errors": old_pr.get("errors", 0),
                                    "best_ratio": old_pr.get("best_ratio", 0.0),
                                    "best_streak": old_pr.get("best_streak", 0)
                                },
                                "non_colored": default_pr_structure() # Add default for non_colored
                            }
                        else:
                            # Ensure both colored and non_colored keys exist
                            if "colored" not in data[atlas]:
                                data[atlas]["colored"] = default_pr_structure()
                            if "non_colored" not in data[atlas]:
                                data[atlas]["non_colored"] = default_pr_structure()
                            # Ensure all sub-keys exist within colored/non_colored
                            for mode in ["colored", "non_colored"]:
                                for key, default_value in default_pr_structure().items():
                                     if key not in data[atlas][mode]:
                                         data[atlas][mode][key] = default_value

                return data
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
             print(f"Warning: Could not load or parse PR file ({e}). Creating default PR data.")
             # Create default structure for all atlases
             return {
                 atlas: {
                     "colored": default_pr_structure(),
                     "non_colored": default_pr_structure()
                 }
                 for atlas in self.atlas_options.keys()
             }


    def save_pr(self):
        """Save personal records to JSON file."""
        try:
            with open(self.pr_file, 'w') as f:
                json.dump(self.pr_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save PR data: {e}")
            # Optionally show a message box to the user
            # QMessageBox.warning(self, "Save Error", f"Could not save personal records:\n{e}")


    def setup_ui(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # ==================================
        # == Landing Page Widget Setup =====
        # ==================================
        self.landing_widget = QWidget()
        landing_layout = QVBoxLayout(self.landing_widget)
        landing_layout.setAlignment(Qt.AlignCenter)
        landing_layout.setSpacing(30)
        landing_layout.setContentsMargins(50, 50, 50, 50) # Add margins

        # --- Logo and Title ---
        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignCenter)

        logo_label = QLabel()
        logo_path = get_resource_path("code/neuroguessr5.png") # Ensure logo is in code folder or adjust path
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            logo_label.setText("Logo\nNot Found")
            logo_label.setFixedSize(300, 300) # Placeholder size
            logo_label.setStyleSheet("border: 1px solid white; color: white; qproperty-alignment: 'AlignCenter';")
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

        # --- Game Mode Selection ---
        mode_label = QLabel("Game Mode")
        mode_label.setStyleSheet("color: white; font-size: 18px;")
        mode_label.setFont(QFont("Helvetica [Cronyx]", 20, QFont.Bold))
        mode_label.setAlignment(Qt.AlignCenter)
        landing_layout.addWidget(mode_label)

        mode_buttons_layout = QHBoxLayout()
        mode_buttons_layout.setSpacing(20)

        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True) # Only one mode selectable

        practice_button = QPushButton("Practice")
        # Add styles, font, checkable, group
        practice_button.setStyleSheet("""
            QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
            QPushButton:checked {background-color: #3E3E42; border: 2px solid #0078D7;}
            QPushButton:hover {background-color: #3E3E42;}
        """)
        practice_button.setFont(QFont("Helvetica [Cronyx]", 16))
        practice_button.setCheckable(True)
        practice_button.setChecked(True) # Default mode
        self.mode_button_group.addButton(practice_button, 0) # ID 0 for Practice

        contre_button = QPushButton()
        contre_button.setText("Contre la Montre   ") # Extra spaces for icon
        contre_icon_path = get_resource_path("code/speedometer.png") # Ensure icon is present
        contre_pixmap = QPixmap(contre_icon_path)
        if not contre_pixmap.isNull():
            contre_button.setIcon(QIcon(contre_pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        # Add styles, font, checkable, group
        contre_button.setStyleSheet(practice_button.styleSheet()) # Reuse style
        contre_button.setFont(QFont("Helvetica [Cronyx]", 16))
        contre_button.setCheckable(True)
        self.mode_button_group.addButton(contre_button, 1) # ID 1 for Contre la Montre

        streak_button = QPushButton()
        streak_button.setText("Streak   ") # Extra spaces for icon
        streak_icon_path = get_resource_path("code/flame.png") # Ensure icon is present
        streak_pixmap = QPixmap(streak_icon_path)
        if not streak_pixmap.isNull():
            streak_button.setIcon(QIcon(streak_pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        # Add styles, font, checkable, group
        streak_button.setStyleSheet(practice_button.styleSheet()) # Reuse style
        streak_button.setFont(QFont("Helvetica [Cronyx]", 16))
        streak_button.setCheckable(True)
        self.mode_button_group.addButton(streak_button, 2) # ID 2 for Streak

        mode_buttons_layout.addWidget(practice_button)
        mode_buttons_layout.addWidget(contre_button)
        mode_buttons_layout.addWidget(streak_button)
        landing_layout.addLayout(mode_buttons_layout)

        # --- Atlas Coloration Selection ---
        atlas_color_label = QLabel("Atlas Coloration")
        atlas_color_label.setStyleSheet("color: white; font-size: 18px;")
        atlas_color_label.setFont(QFont("Helvetica [Cronyx]", 20, QFont.Bold))
        atlas_color_label.setAlignment(Qt.AlignCenter)
        landing_layout.addWidget(atlas_color_label)

        color_buttons_layout = QHBoxLayout()
        color_buttons_layout.setSpacing(20)

        self.color_button_group = QButtonGroup(self)
        self.color_button_group.setExclusive(True)

        colored_button = QPushButton("Colored Atlas")
        # Add styles, font, checkable, group
        colored_button.setStyleSheet("""
            QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
            QPushButton:checked {background-color: #3E3E42; border: 2px solid #4CAF50;} /* Green border when checked */
            QPushButton:hover {background-color: #3E3E42;}
        """)
        colored_button.setFont(QFont("Helvetica [Cronyx]", 16))
        colored_button.setCheckable(True)
        colored_button.setChecked(True) # Default colored
        self.color_button_group.addButton(colored_button, 0) # ID 0 for Colored

        non_colored_button = QPushButton("No Colors")
        # Add styles, font, checkable, group
        non_colored_button.setStyleSheet("""
            QPushButton {background-color: #2D2D30; color: white; border: 2px solid #444; border-radius: 10px; padding: 15px; font-size: 16px;}
            QPushButton:checked {background-color: #3E3E42; border: 2px solid #f44336;} /* Red border when checked */
            QPushButton:hover {background-color: #3E3E42;}
        """)
        non_colored_button.setFont(QFont("Helvetica [Cronyx]", 16))
        non_colored_button.setCheckable(True)
        self.color_button_group.addButton(non_colored_button, 1) # ID 1 for Non-colored

        color_buttons_layout.addWidget(colored_button)
        color_buttons_layout.addWidget(non_colored_button)
        landing_layout.addLayout(color_buttons_layout)

        # --- Atlas Selection ---
        atlas_label = QLabel("Select Atlas")
        atlas_label.setStyleSheet("color: white; font-size: 18px;")
        atlas_label.setFont(QFont("Helvetica [Cronyx]", 20, QFont.Bold))
        atlas_label.setAlignment(Qt.AlignCenter)
        landing_layout.addWidget(atlas_label)

        atlas_buttons_layout = QGridLayout()
        atlas_buttons_layout.setSpacing(10)

        self.atlas_button_group = QButtonGroup(self)
        self.atlas_button_group.setExclusive(True)

        atlas_names = list(self.atlas_options.keys())
        for i, atlas_name in enumerate(atlas_names):
            atlas_button = QPushButton(atlas_name)
            # Add styles, font, checkable, group
            atlas_button.setStyleSheet(practice_button.styleSheet()) # Reuse style
            atlas_button.setFont(QFont("Helvetica [Cronyx]", 16))
            atlas_button.setCheckable(True)
            if atlas_name == self.current_atlas: # Check the default atlas
                atlas_button.setChecked(True)
            self.atlas_button_group.addButton(atlas_button, i) # Use index as ID
            row = i // 3 # Arrange in a grid (3 columns)
            col = i % 3
            atlas_buttons_layout.addWidget(atlas_button, row, col)

        landing_layout.addLayout(atlas_buttons_layout)

        # --- Personal Best Display ---
        self.pr_box = QGroupBox("Personal Best")
        self.pr_box.setStyleSheet("""
            QGroupBox {color: white; font-size: 18px; font-weight: bold; border: 2px solid #444; border-radius: 10px; padding-top: 25px; margin-top: 10px;} /* Adjusted padding/margin */
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px;} /* Center title */
        """)
        self.pr_box.setFont(QFont("Helvetica [Cronyx]", 20, QFont.Bold))
        self.pr_box.setAlignment(Qt.AlignCenter) # Align title
        pr_layout = QVBoxLayout(self.pr_box)
        pr_layout.setSpacing(15) # Reduced spacing
        pr_layout.setAlignment(Qt.AlignCenter)

        # PR - Accuracy (Ratio)
        ratio_layout = QHBoxLayout()
        ratio_layout.setAlignment(Qt.AlignCenter)
        ratio_icon = QLabel()
        ratio_icon_path = get_resource_path("code/speedometer.png") # Check path
        ratio_pixmap = QPixmap(ratio_icon_path)
        if not ratio_pixmap.isNull():
             ratio_icon.setPixmap(ratio_pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: ratio_icon.setText("🎯") # Fallback emoji
        ratio_layout.addWidget(ratio_icon)
        self.ratio_pr_label = QLabel("Accuracy: N/A")
        self.ratio_pr_label.setStyleSheet("color: white; font-size: 18px;")
        self.ratio_pr_label.setFont(QFont("Helvetica [Cronyx]", 22))
        ratio_layout.addWidget(self.ratio_pr_label)
        ratio_layout.addStretch() # Push content to center
        pr_layout.addLayout(ratio_layout)

        # PR - Time
        time_layout = QHBoxLayout()
        time_layout.setAlignment(Qt.AlignCenter)
        time_icon = QLabel()
        time_icon_path = get_resource_path("code/stopwatch.png") # Check path
        time_pixmap = QPixmap(time_icon_path)
        if not time_pixmap.isNull():
            time_icon.setPixmap(time_pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: time_icon.setText("⏱️") # Fallback emoji
        time_layout.addWidget(time_icon)
        self.time_pr_label = QLabel("Time: N/A")
        self.time_pr_label.setStyleSheet("color: white; font-size: 18px;")
        self.time_pr_label.setFont(QFont("Helvetica [Cronyx]", 22))
        time_layout.addWidget(self.time_pr_label)
        time_layout.addStretch()
        pr_layout.addLayout(time_layout)

        # PR - Streak
        streak_layout = QHBoxLayout()
        streak_layout.setAlignment(Qt.AlignCenter)
        streak_icon = QLabel()
        streak_icon_path = get_resource_path("code/flame.png") # Check path
        streak_pixmap = QPixmap(streak_icon_path)
        if not streak_pixmap.isNull():
             streak_icon.setPixmap(streak_pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: streak_icon.setText("🔥") # Fallback emoji
        streak_layout.addWidget(streak_icon)
        self.streak_pr_label = QLabel("Streak: N/A")
        self.streak_pr_label.setStyleSheet("color: white; font-size: 18px;")
        self.streak_pr_label.setFont(QFont("Helvetica [Cronyx]", 22))
        streak_layout.addWidget(self.streak_pr_label)
        streak_layout.addStretch()
        pr_layout.addLayout(streak_layout)

        landing_layout.addWidget(self.pr_box)

        # Connect signals to update PR display
        self.update_pr_label() # Initial update
        self.atlas_button_group.buttonClicked.connect(self.update_pr_label)
        self.color_button_group.buttonClicked.connect(self.update_pr_label)

        # --- Quit and Play Buttons ---
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)

        quit_button = QPushButton("Quit")
        # Add styles, font, connect signal
        quit_button.setStyleSheet("""
            QPushButton {font-size: 18px; padding: 15px 40px; background-color: #f44336; color: white; border-radius: 10px; border: none; font-weight: bold;}
            QPushButton:hover {background-color: #d32f2f;}
        """)
        quit_button.setFont(QFont("Helvetica [Cronyx]", 18, QFont.Bold))
        quit_button.clicked.connect(QApplication.instance().quit)

        play_button = QPushButton("Play")
        # Add styles, font, connect signal
        play_button.setStyleSheet("""
            QPushButton {font-size: 18px; padding: 15px 40px; background-color: #4CAF50; color: white; border-radius: 10px; border: none; font-weight: bold;}
            QPushButton:hover {background-color: #45a049;}
        """)
        play_button.setFont(QFont("Helvetica [Cronyx]", 18, QFont.Bold))
        play_button.clicked.connect(self.start_game_from_landing)

        buttons_layout.addWidget(quit_button)
        buttons_layout.addWidget(play_button)
        landing_layout.addLayout(buttons_layout)
        landing_layout.addStretch() # Push everything up

        self.stacked_widget.addWidget(self.landing_widget)

        # ==================================
        # ===== Game Widget Setup ==========
        # ==================================
        self.game_widget = QWidget()
        main_game_layout = QHBoxLayout(self.game_widget)

        # --- Left Panel (Controls & Views) ---
        left_panel = QWidget()
        game_layout = QVBoxLayout(left_panel)

        # Top Controls (Atlas Info, Visibility Toggle, Template Switch)
        top_controls_layout = QHBoxLayout()

        # Atlas Info Box
        atlas_info_box = QGroupBox("Info")
        atlas_info_box.setStyleSheet("QGroupBox { color: white; border: 1px solid #555; border-radius: 5px; margin-top: 1ex; } "
                                     "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        atlas_info_layout = QHBoxLayout(atlas_info_box)
        atlas_label = QLabel("Atlas:")
        atlas_label.setStyleSheet("color: white; border: none;")
        atlas_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.active_atlas_label = QLabel(self.current_atlas)
        self.active_atlas_label.setStyleSheet("color: white; font-weight: bold; border: none;")
        self.active_atlas_label.setFont(QFont("Helvetica [Cronyx]", 12, QFont.Bold))
        atlas_info_layout.addWidget(atlas_label)
        atlas_info_layout.addWidget(self.active_atlas_label)

        # Template Switch Button
        self.template_switch_button = QPushButton(" MNI ↔ BigBrain ")
        self.template_switch_button.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 5px 10px; background-color: #2E7D32; /* Darker Green */
                color: white; border-radius: 5px; border: 1px solid #66BB6A; /* Lighter Green Border */
                font-weight: bold;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:pressed { background-color: #1B5E20; }
        """)
        self.template_switch_button.setFont(QFont("Helvetica [Cronyx]", 12, QFont.Bold))
        self.template_switch_button.setToolTip("Switch between MNI and BigBrain templates")
        self.template_switch_button.clicked.connect(self.switch_template)


        # Atlas Visibility Toggle
        self.atlas_toggle = QCheckBox("Show Atlas Regions")
        self.atlas_toggle.setChecked(True)
        self.atlas_toggle.setStyleSheet("color: white; font-size: 12px; border: none;")
        self.atlas_toggle.setFont(QFont("Helvetica [Cronyx]", 12))
        self.atlas_toggle.stateChanged.connect(self.toggle_atlas_visibility)

        # Arrange Top Controls
        top_controls_layout.addWidget(atlas_info_box)
        top_controls_layout.addWidget(self.template_switch_button)
        top_controls_layout.addStretch(1) # Add stretch to push toggle right
        top_controls_layout.addWidget(self.atlas_toggle)
        game_layout.addLayout(top_controls_layout)


        # Status Bar (Target, Timer, Score/Errors)
        status_layout = QHBoxLayout()
        self.target_label = QLabel("Target: Not Started")
        self.target_label.setFont(QFont("Helvetica [Cronyx]", 18, QFont.Bold)) # Slightly smaller
        self.target_label.setStyleSheet("color: white; background-color: #333; padding: 8px; border-radius: 5px;")
        self.target_label.setAlignment(Qt.AlignCenter)

        self.timer_label = QLabel("Time: N/A")
        self.timer_label.setFont(QFont("Helvetica [Cronyx]", 14))
        self.timer_label.setStyleSheet("color: white; background-color: #333; padding: 8px; border-radius: 5px;")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setMinimumWidth(120) # Ensure space for time display

        score_box = QGroupBox("Stats")
        score_box.setStyleSheet("QGroupBox { color: white; border: 1px solid #555; border-radius: 5px; margin-top: 1ex; } "
                                "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        score_layout = QVBoxLayout(score_box)
        self.score_label = QLabel("Correct: 0")
        self.score_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.score_label.setStyleSheet("color: white; border: none;")
        self.error_label = QLabel("Errors: 0")
        self.error_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.error_label.setStyleSheet("color: white; border: none;")
        score_layout.addWidget(self.score_label)
        score_layout.addWidget(self.error_label)
        score_layout.addStretch() # Push labels up

        status_layout.addWidget(self.target_label, 3) # Target takes more space
        status_layout.addWidget(self.timer_label, 1)
        status_layout.addWidget(score_box, 1)
        game_layout.addLayout(status_layout)

        # Brain Slice Views
        views_layout = QHBoxLayout()
        self.slice_views = []
        for i in range(3):
            view = BrainSliceView(i)
            view.slice_clicked.connect(self.handle_slice_click)
            view.slice_changed.connect(self.handle_slice_change)
            self.slice_views.append(view)
            views_layout.addWidget(view)
        game_layout.addLayout(views_layout, 1) # Views take expanding space

        # Sliders
        slider_layout = QHBoxLayout()
        # Z (Axial)
        z_layout = QVBoxLayout()
        # z_label = QLabel("Axial") # Title already in view
        # z_label.setAlignment(Qt.AlignCenter)
        # z_label.setStyleSheet("color: white;")
        # z_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(100) # Will be set dynamically
        self.z_slider.setValue(50)
        self.z_slider.valueChanged.connect(lambda v, p=0: self.update_slice_position(p, v)) # Use lambda with default arg
        # z_layout.addWidget(z_label)
        z_layout.addWidget(self.z_slider)
        # Y (Coronal)
        y_layout = QVBoxLayout()
        # y_label = QLabel("Coronal")
        # y_label.setAlignment(Qt.AlignCenter)
        # y_label.setStyleSheet("color: white;")
        # y_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.setMaximum(100)
        self.y_slider.setValue(50)
        self.y_slider.valueChanged.connect(lambda v, p=1: self.update_slice_position(p, v))
        # y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_slider)
        # X (Sagittal)
        x_layout = QVBoxLayout()
        # x_label = QLabel("Sagittal")
        # x_label.setAlignment(Qt.AlignCenter)
        # x_label.setStyleSheet("color: white;")
        # x_label.setFont(QFont("Helvetica [Cronyx]", 12))
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(100)
        self.x_slider.setValue(50)
        self.x_slider.valueChanged.connect(lambda v, p=2: self.update_slice_position(p, v))
        # x_layout.addWidget(x_label)
        x_layout.addWidget(self.x_slider)

        slider_layout.addLayout(z_layout)
        slider_layout.addLayout(y_layout)
        slider_layout.addLayout(x_layout)
        game_layout.addLayout(slider_layout)

        # Bottom Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Game")
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;}
            QPushButton:hover {background-color: #45a049;}
        """)
        self.start_button.setFont(QFont("Helvetica [Cronyx]", 16))

        self.guess_button = QPushButton("Confirm Guess (Space)")
        self.guess_button.clicked.connect(self.validate_guess)
        self.guess_button.setEnabled(False)
        self.guess_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #2196F3; color: white; border-radius: 5px; font-weight: bold;}
            QPushButton:disabled {background-color: #cccccc; color: #666666;}
            QPushButton:hover:enabled {background-color: #1976D2;}
        """)
        self.guess_button.setFont(QFont("Helvetica [Cronyx]", 16, QFont.Bold))
        self.guess_button.setShortcut(Qt.Key_Space) # Add Space shortcut

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        self.help_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #FFC107; color: black; border-radius: 5px;} /* Amber */
            QPushButton:hover {background-color: #FFA000;}
        """)
        self.help_button.setFont(QFont("Helvetica [Cronyx]", 16))

        self.menu_button = QPushButton("Menu")
        self.menu_button.clicked.connect(self.show_menu)
        self.menu_button.setStyleSheet("""
            QPushButton {font-size: 16px; padding: 10px; background-color: #FF9800; color: white; border-radius: 5px;} /* Orange */
            QPushButton:hover {background-color: #e68a00;}
        """)
        self.menu_button.setFont(QFont("Helvetica [Cronyx]", 16))

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.guess_button)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.menu_button)
        game_layout.addLayout(button_layout)

        self.guess_button.hide() # Hide guess button initially

        # --- Right Panel (Memo/Info) ---
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
        # self.memo_text.setMinimumWidth(300) # Allow it to resize
        self.memo_text.setMaximumWidth(450) # Limit max width
        memo_layout.addWidget(self.memo_text)
        memo_layout.addStretch()

        # Add panels to main game layout
        main_game_layout.addWidget(left_panel, 3) # Left panel takes more space
        main_game_layout.addWidget(self.memo_widget, 1)

        # --- Final Setup ---
        self.setFocusPolicy(Qt.StrongFocus) # Allow main window to receive key events
        # self.keyPressEvent defined below

        self.stacked_widget.addWidget(self.game_widget)
        self.stacked_widget.setCurrentWidget(self.landing_widget) # Start on landing page
        # self.load_data() # Load data when game starts, not on init

    def update_pr_label(self):
        """Updates the Personal Record labels on the landing page based on selection."""
        atlas_names = list(self.atlas_options.keys())
        selected_atlas_id = self.atlas_button_group.checkedId()
        if selected_atlas_id < 0 or selected_atlas_id >= len(atlas_names):
             print("Warning: Invalid atlas selection ID for PR.")
             atlas = atlas_names[0] # Default to first atlas if error
        else:
            atlas = atlas_names[selected_atlas_id]

        color_mode = "colored" if self.color_button_group.checkedId() == 0 else "non_colored"

        # Safely get PR data, providing default if atlas or mode is missing
        default_pr = {"time": float("inf"), "errors": 0, "best_ratio": 0.0, "best_streak": 0}
        atlas_pr = self.pr_data.get(atlas, {"colored": default_pr, "non_colored": default_pr})
        pr = atlas_pr.get(color_mode, default_pr)


        # Update Time PR
        if pr.get("time", float("inf")) == float("inf"):
             self.time_pr_label.setText("Time: N/A")
        else:
             time_val = pr["time"]
             minutes = int(time_val // 60)
             seconds = int(time_val % 60)
             self.time_pr_label.setText(f"Time: {minutes}'{seconds:02d}\"") # Format M'SS"

        # Update Accuracy PR
        best_ratio = pr.get("best_ratio", 0.0)
        self.ratio_pr_label.setText(f"Accuracy: {best_ratio:.1f}%" if best_ratio > 0 else "Accuracy: N/A")

        # Update Streak PR
        best_streak = pr.get("best_streak", 0)
        self.streak_pr_label.setText(f"Streak: {best_streak}" if best_streak > 0 else "Streak: N/A")


    def toggle_atlas_visibility(self, state):
        """Toggles the visibility of the colored atlas overlay."""
        self.show_atlas = (state == Qt.Checked)
        self.update_all_slices() # Redraw slices with/without overlay


    def switch_template(self):
        """Switches between MNI and BigBrain templates."""
        if self.current_template_name == "MNI":
            self.current_template_name = "BigBrain"
            bg_color = "white"
        else:
            self.current_template_name = "MNI"
            bg_color = "black"

        print(f"Switching template to: {self.current_template_name}")

        # Update button text/tooltip (optional)
        # self.template_switch_button.setText(f" {self.current_template_name} ↔ {'BigBrain' if self.current_template_name == 'MNI' else 'MNI'} ")

        # Reload only the template data
        if not self.load_data(load_atlas=False):
             # Handle error if template loading fails (e.g., file not found)
             # Revert the switch
             if self.current_template_name == "MNI":
                 self.current_template_name = "BigBrain"
                 bg_color = "white"
             else:
                 self.current_template_name = "MNI"
                 bg_color = "black"
             print("Template switch reverted due to loading error.")
             return # Stop further processing


        # Update background color of slice views
        for view in self.slice_views:
            view.set_background(bg_color)

        # Update the slices immediately
        self.update_all_slices()


    def reset_game_ui(self):
        """Resets the game UI elements to their initial state before a game starts."""
        self.start_button.show()
        self.guess_button.hide()
        self.target_label.setText("Target: Not Started")
        self.score_label.setText("Correct: 0")
        self.error_label.setText("Errors: 0")

        # Reset timer based on mode
        if self.game_mode == "Contre la Montre":
            self.timer_label.setText("Time: 0'00\"")
        elif self.game_mode == "Streak":
             self.timer_label.setText("Time: N/A")
             self.score_label.setText("Streak: 0") # Reset streak label too
        else: # Practice
            self.timer_label.setText("Time: N/A")

        self.guess_button.setEnabled(False)
        self.guess_button.setText("Confirm Guess (Space)") # Reset button text

        # --- Reset Game State ---
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
        # self.time_remaining = 180 # Should be reset when starting Contre la Montre specifically
        self.game_running = False

        # --- Reset Visuals ---
        self.show_atlas = self.use_colored_atlas # Reset visibility based on landing page choice
        self.atlas_toggle.setChecked(self.show_atlas)

        # Reset template to default (MNI) and background color
        self.current_template_name = "MNI"
        bg_color = "black"
        if not self.load_data(load_atlas=False): # Ensure default template is loaded
             print("Warning: Failed to reload default MNI template during UI reset.")
        for view in self.slice_views:
             view.stop_blinking() # Stop any blinking from previous game
             view.set_background(bg_color) # Set default background


        # Reset memo visibility based on mode
        self.memo_widget.setVisible(self.game_mode == "Practice")
        self.memo_text.clear() # Clear previous info

        # Reset sliders to center (optional, or keep last position)
        # if self.brain_data:
        #     self.z_slider.setValue(self.brain_data.shape[2] // 2)
        #     self.y_slider.setValue(self.brain_data.shape[1] // 2)
        #     self.x_slider.setValue(self.brain_data.shape[0] // 2)
        # else:
        #     self.z_slider.setValue(50)
        #     self.y_slider.setValue(50)
        #     self.x_slider.setValue(50)

        # Ensure focus is reasonable (e.g., back on the main widget)
        self.setFocus()


    def handle_slice_change(self, plane_index, delta):
        """Handles slice changes triggered by wheel events on BrainSliceView."""
        if plane_index == 0: # Axial
            slider = self.z_slider
        elif plane_index == 1: # Coronal
            slider = self.y_slider
        else: # Sagittal
            slider = self.x_slider

        new_value = slider.value() + delta
        # Clamp value to slider range
        new_value = max(slider.minimum(), min(slider.maximum(), new_value))
        # Setting the slider value will trigger update_slice_position via its signal
        slider.setValue(new_value)


    def start_game_from_landing(self):
        """Sets up game parameters based on landing page choices and switches to game view."""
        # --- Determine Game Mode ---
        selected_mode_id = self.mode_button_group.checkedId()
        if selected_mode_id == 0:
            self.game_mode = "Practice"
        elif selected_mode_id == 1:
            self.game_mode = "Contre la Montre"
        else: # ID 2
            self.game_mode = "Streak"

        # --- Determine Atlas ---
        selected_atlas_id = self.atlas_button_group.checkedId()
        atlas_names = list(self.atlas_options.keys())
        if 0 <= selected_atlas_id < len(atlas_names):
            self.current_atlas = atlas_names[selected_atlas_id]
        else:
            QMessageBox.warning(self, "Atlas Error", "Invalid atlas selected. Defaulting to AAL.")
            self.current_atlas = "AAL"
            self.atlas_button_group.button(atlas_names.index("AAL")).setChecked(True)

        # --- Determine Coloration ---
        self.use_colored_atlas = self.color_button_group.checkedId() == 0
        self.show_atlas = self.use_colored_atlas # Initial visibility matches choice

        # --- Update UI Elements ---
        self.active_atlas_label.setText(self.current_atlas)
        self.atlas_toggle.setChecked(self.show_atlas)

        # --- Load Data (Atlas and Default Template) ---
        self.current_template_name = "MNI" # Ensure default template is selected
        if not self.load_data(load_atlas=True):
            QMessageBox.critical(self, "Loading Error", "Failed to load essential game data. Cannot start game.")
            return # Don't proceed if data loading fails

        # --- Set Initial Background Color Based on Default Template ---
        initial_bg_color = "black" # MNI is default
        for view in self.slice_views:
             view.set_background(initial_bg_color)
        self.update_all_slices() # Display initial slices

        # --- Configure UI for Selected Mode ---
        self.set_game_mode(self.game_mode) # Configures timer, labels, memo visibility
        self.reset_game_ui() # Reset scores, timers, buttons etc.
        # self.update_pr_label() # PR label is on landing page, no need to update here

        # --- Switch View ---
        self.stacked_widget.setCurrentWidget(self.game_widget)
        self.game_widget.setFocus() # Set focus to game widget for keyboard events


    def keyPressEvent(self, event):
        """Handle global key press events, specifically Space for guessing."""
        if event.key() == Qt.Key_Space and self.game_running and self.guess_button.isEnabled() and self.guess_button.isVisible():
            self.validate_guess()
        else:
             # Allow other key events (like arrow keys for sliders if they had focus) to be processed
             super().keyPressEvent(event)


    def load_data(self, load_atlas=True):
        """
        Loads the NIfTI data for the selected atlas and/or template.
        Returns True on success, False on critical failure.
        """
        atlas_name = self.current_atlas
        template_name = self.current_template_name

        # --- Load Template Data ---
        template_file = self.template_files.get(template_name)
        if not template_file or not os.path.exists(template_file):
            QMessageBox.critical(self, "Template Error", f"Template file not found for '{template_name}':\n{template_file}\nCannot continue.")
            return False # Critical error

        try:
            print(f"Loading template: {template_file}")
            self.template_data = nib.load(template_file)
            print(f"Template '{template_name}' loaded successfully. Shape: {self.template_data.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Template Load Error", f"Failed to load template file '{template_name}':\n{str(e)}\nCannot continue.")
            # self.load_dummy_data() # Or just fail
            return False # Critical error

        # --- Load Atlas Data (if requested) ---
        if load_atlas:
            atlas_file, region_file = self.atlas_options.get(atlas_name, (None, None))
            json_file = None
            if region_file:
                 json_file = os.path.splitext(region_file)[0] + ".json"

            if not atlas_file or not region_file or not os.path.exists(atlas_file) or not os.path.exists(region_file):
                 error_msg = ""
                 if not atlas_file or not os.path.exists(atlas_file): error_msg += f"Atlas file not found or invalid: {atlas_file}\n"
                 if not region_file or not os.path.exists(region_file): error_msg += f"Region file not found or invalid: {region_file}\n"
                 QMessageBox.critical(self, "Atlas Error", f"Essential files missing for atlas '{atlas_name}':\n{error_msg}Cannot load atlas.")
                 # Keep the previously loaded atlas if available? Or load dummy? For now, fail.
                 # self.load_dummy_data()
                 return False # Consider if this is critical, maybe allow playing with just template?

            try:
                print(f"Loading atlas: {atlas_file}")
                self.brain_data = nib.load(atlas_file)
                print(f"Atlas '{atlas_name}' loaded successfully. Shape: {self.brain_data.shape}")

                # Check for shape mismatch between loaded atlas and template
                if self.brain_data.shape != self.template_data.shape:
                    QMessageBox.warning(self, "Shape Mismatch",
                                          f"Warning: Atlas '{atlas_name}' shape {self.brain_data.shape} "
                                          f"does not match template '{template_name}' shape {self.template_data.shape}. "
                                          "Display might be incorrect.")
                    # Decide how to handle: proceed with warning, try resizing (complex), or fail?
                    # For now, proceed with warning.

                # Load region names and colors
                print(f"Loading regions: {region_file}")
                # Handle potential variations in separator (space or tab)
                try:
                    region_df = pd.read_csv(region_file, sep=r'\s+', comment="#", header=None,
                                            names=["Index", "RegionName", "R", "G", "B", "A"], engine='python')
                except pd.errors.ParserError:
                     # Try with tab separator if space fails
                     region_df = pd.read_csv(region_file, sep='\t', comment="#", header=None,
                                            names=["Index", "RegionName", "R", "G", "B", "A"], engine='python')

                # Convert relevant columns to appropriate types
                region_df['Index'] = region_df['Index'].astype(int)
                region_df['R'] = region_df['R'].astype(int)
                region_df['G'] = region_df['G'].astype(int)
                region_df['B'] = region_df['B'].astype(int)

                # Create mappings
                self.region_map = {row["Index"]: row["RegionName"] for _, row in region_df.iterrows()}
                self.colormap = {row["Index"]: (row["R"], row["G"], row["B"]) for _, row in region_df.iterrows()}
                print(f"Loaded {len(self.region_map)} regions for '{atlas_name}'.")


                # Load region info (JSON) if available
                self.region_info = {} # Reset info
                if json_file and os.path.exists(json_file):
                    try:
                        print(f"Loading region info: {json_file}")
                        with open(json_file, 'r', encoding='utf-8') as f: # Specify encoding
                            self.region_info = json.load(f)
                        print(f"Loaded info for {len(self.region_info)} regions.")
                    except Exception as e_json:
                        print(f"Warning: Failed to load or parse JSON file {json_file}: {e_json}")
                        QMessageBox.warning(self, "Info Load Warning", f"Could not load region information from:\n{json_file}\n\nError: {e_json}")
                else:
                    print(f"Warning: JSON file not found or not specified for region info: {json_file}")


                # --- Update UI Elements Dependent on Data Shape ---
                # Set slider ranges based on the *atlas* data shape
                max_x, max_y, max_z = self.brain_data.shape
                self.x_slider.setMaximum(max_x - 1)
                self.y_slider.setMaximum(max_y - 1)
                self.z_slider.setMaximum(max_z - 1)

                # Set initial slider positions (center of the volume)
                initial_x = max_x // 2
                initial_y = max_y // 2
                initial_z = max_z // 2
                # Block signals temporarily to avoid triggering updates multiple times
                self.x_slider.blockSignals(True)
                self.y_slider.blockSignals(True)
                self.z_slider.blockSignals(True)
                self.x_slider.setValue(initial_x)
                self.y_slider.setValue(initial_y)
                self.z_slider.setValue(initial_z)
                self.x_slider.blockSignals(False)
                self.y_slider.blockSignals(False)
                self.z_slider.blockSignals(False)

                # Set initial 3D crosshair and positions
                self.current_positions = [initial_z, initial_y, initial_x]
                self.crosshair_3d = (initial_x, initial_y, initial_z)

                # # Update slices immediately after loading new atlas
                # self.update_all_slices() # This might be called again later, but good to have initial view

            except Exception as e:
                QMessageBox.critical(self, "Atlas Load Error", f"Failed to load atlas data for '{atlas_name}':\n{str(e)}\nCannot load atlas.")
                # self.load_dummy_data() # Or fail completely
                return False # Critical error if atlas loading fails

        # If we reach here, loading was successful (or only template was loaded successfully)
        return True


    def load_dummy_data(self):
        """Loads placeholder data if primary data loading fails."""
        print("Loading dummy data...")
        dummy_shape = (100, 100, 100) # Smaller dummy shape
        # Create dummy atlas data with distinct regions
        dummy_data = np.zeros(dummy_shape, dtype=np.int16)
        dummy_data[10:40, 10:40, 10:40] = 1 # Region 1
        dummy_data[60:90, 10:40, 10:40] = 2 # Region 2
        dummy_data[10:40, 60:90, 10:40] = 3 # Region 3
        dummy_data[10:40, 10:40, 60:90] = 4 # Region 4
        dummy_data[60:90, 60:90, 60:90] = 5 # Region 5

        # Create dummy template data (e.g., gradient)
        x, y, z = np.indices(dummy_shape)
        dummy_template = (x + y + z).astype(np.float32)

        # Define dummy regions and colormap
        regions = {
            1: "Dummy Region Alpha", 2: "Dummy Region Beta", 3: "Dummy Gamma",
            4: "Dummy Delta", 5: "Dummy Epsilon"
        }
        self.colormap = {
            1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
            4: (255, 255, 0), 5: (0, 255, 255)
        }
        # Dummy region info
        self.region_info = {
            "1": {"name": "Dummy Region Alpha", "structure": ["Made of voxels."], "function": ["Does dummy things."]},
            "2": {"name": "Dummy Region Beta", "structure": ["Also voxels."], "function": ["Beta functions."]},
             # Add info for others if needed
        }

        # Create Nifti objects
        affine = np.eye(4) # Simple identity affine
        self.brain_data = nib.Nifti1Image(dummy_data, affine)
        self.template_data = nib.Nifti1Image(dummy_template, affine)
        self.region_map = regions

        # Update sliders for dummy data shape
        max_x, max_y, max_z = dummy_shape
        self.x_slider.setMaximum(max_x - 1)
        self.y_slider.setMaximum(max_y - 1)
        self.z_slider.setMaximum(max_z - 1)
        self.x_slider.setValue(max_x // 2)
        self.y_slider.setValue(max_y // 2)
        self.z_slider.setValue(max_z // 2)

        # Set initial positions and update display
        self.current_positions = [max_z // 2, max_y // 2, max_x // 2]
        self.crosshair_3d = (max_x // 2, max_y // 2, max_z // 2)
        self.update_all_slices()
        print("Dummy data loaded.")


    def set_game_mode(self, mode):
        """Configures UI elements based on the selected game mode."""
        self.game_mode = mode
        print(f"Setting game mode to: {mode}")

        if mode == "Practice":
            self.timer_label.setText("Time: N/A")
            self.score_label.setText("Correct: 0")
            self.error_label.setText("Errors: 0")
            self.memo_widget.setVisible(True) # Show info panel
            self.target_label.setText("Target: Select Region") # Initial prompt

        elif mode == "Contre la Montre":
            self.timer_label.setText("Time: 0'00\"") # Initial time display
            self.score_label.setText("Found: 0 / ?") # Placeholder until regions loaded
            self.error_label.setText("Errors: 0")
            self.memo_widget.setVisible(False) # Hide info panel
            self.target_label.setText("Target: Loading...")

        elif mode == "Streak":
            self.timer_label.setText("Time: N/A")
            self.score_label.setText("Streak: 0")
            self.error_label.setText("Errors: 0") # Errors still tracked internally, but not primary display
            self.memo_widget.setVisible(False) # Hide info panel
            self.target_label.setText("Target: Loading...")

        else:
             print(f"Warning: Unknown game mode '{mode}' selected.")


    def start_game(self):
        """Starts the selected game mode, initializing scores, timers, and targets."""
        if not self.brain_data or not self.template_data or not self.region_map:
             QMessageBox.critical(self, "Start Error", "Cannot start game. Essential data is missing.")
             return

        # --- Reset Core Game State ---
        self.score = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.correct_guesses = []
        self.incorrect_guesses = []
        self.streak_guessed_regions = []
        self.selected_position = None # Clear last click position
        self.game_running = True

        # --- Mode-Specific Setup ---
        if self.game_mode == "Contre la Montre":
            # Get all valid, non-zero regions from the loaded atlas data
            unique_ids = np.unique(self.brain_data.get_fdata().astype(np.int32))
            self.all_regions = [int(val) for val in unique_ids if val > 0 and val in self.region_map]

            if not self.all_regions:
                QMessageBox.critical(self, "Start Error", f"No valid regions found in the selected atlas '{self.current_atlas}'. Cannot start 'Contre la Montre'.")
                self.game_running = False
                return

            self.remaining_regions = self.all_regions.copy()
            random.shuffle(self.remaining_regions) # Randomize the order
            self.start_time = time.time() # Start timer
            self.total_time = 0
            self.score_label.setText(f"Found: 0 / {len(self.all_regions)}")
            self.timer_label.setText("Time: 0'00\"") # Reset display
            self.game_timer.start(1000) # Start the 1-second timer

        elif self.game_mode == "Streak":
            self.score_label.setText("Streak: 0")
            self.error_label.setText("Errors: 0") # Reset error display too
            self.timer_label.setText("Time: N/A")

        elif self.game_mode == "Practice":
            self.score_label.setText("Correct: 0")
            self.error_label.setText("Errors: 0")
            self.timer_label.setText("Time: N/A")

        # --- UI Updates ---
        self.start_button.hide()
        self.guess_button.show()
        self.guess_button.setEnabled(False) # Disable until a click occurs
        self.guess_button.setText("Confirm Guess (Space)")
        self.menu_button.setEnabled(True) # Ensure menu button is active
        for view in self.slice_views:
             view.stop_blinking() # Ensure no blinking at start

        # --- Select First Target ---
        self.select_new_target() # This also updates the target label and memo

        print(f"Game started in '{self.game_mode}' mode.")
        self.setFocus() # Ensure keyboard focus


    def update_timer_display(self):
        """Updates the timer label (used by the QTimer callback)."""
        # This function seems redundant now as update_timer does the work.
        # Kept for potential future use or if separation is desired.
        minutes = self.total_time // 60
        seconds = self.total_time % 60
        self.timer_label.setText(f"Time: {minutes}'{seconds:02d}\"")


    def update_memo_content(self):
        """Updates the information displayed in the memo panel for the current target."""
        if not self.current_target or self.game_mode != "Practice" or not self.memo_widget.isVisible():
            self.memo_text.clear()
            return

        region_id_str = str(self.current_target) # JSON keys are strings
        region_name = self.region_map.get(self.current_target, "Unknown Region")

        if region_id_str in self.region_info:
            info = self.region_info[region_id_str]
            # Use HTML for better formatting
            content = f"<h2 style='margin-bottom: 10px; color: #4CAF50;'>{info.get('name', region_name)}</h2>" # Green title

            if 'structure' in info and info['structure']:
                content += "<h3 style='margin-bottom: 5px; color: #FFC107;'>Structure:</h3>" # Amber subtitle
                content += "<ul style='line-height: 1.6; margin-bottom: 15px; margin-left: 0px; padding-left: 20px;'>" # Indent list
                for item in info['structure']:
                    content += f"<li>{item}</li>"
                content += "</ul>"
            else:
                content += "<p><i>Structure information not available.</i></p>"


            if 'function' in info and info['function']:
                content += "<h3 style='margin-bottom: 5px; color: #2196F3;'>Function:</h3>" # Blue subtitle
                content += "<ul style='line-height: 1.6; margin-bottom: 15px; margin-left: 0px; padding-left: 20px;'>" # Indent list
                for item in info['function']:
                    content += f"<li>{item}</li>"
                content += "</ul>"
            else:
                 content += "<p><i>Function information not available.</i></p>"

            self.memo_text.setHtml(content)
        else:
            # Basic info if JSON details are missing
             content = f"<h2 style='margin-bottom: 10px; color: #4CAF50;'>{region_name}</h2>"
             content += "<p><i>Detailed structure and function information not available in the JSON file.</i></p>"
             self.memo_text.setHtml(content)


    def update_slice_position(self, plane_index, value):
        """Updates the slice index for a given plane and redraws all slices."""
        if not self.brain_data or not self.template_data: return # Exit if no data

        # Update the logical position for the changed plane
        self.current_positions[plane_index] = value

        # Update the 3D crosshair position based on the *active* plane's change
        # This assumes the sliders directly control the coordinates shown in the crosshair
        # current_x, current_y, current_z = self.crosshair_3d
        # if plane_index == 0: # Axial slider controls Z
        #     self.crosshair_3d = (current_x, current_y, value)
        # elif plane_index == 1: # Coronal slider controls Y
        #     self.crosshair_3d = (current_x, value, current_z)
        # elif plane_index == 2: # Sagittal slider controls X
        #     self.crosshair_3d = (value, current_y, current_z)

        # --- Alternative Logic: Sliders control the slice index, crosshair is independent ---
        # Keep self.current_positions updated by sliders.
        # Keep self.crosshair_3d updated ONLY by clicks (handle_slice_click).
        # Slices are displayed based on self.current_positions.
        # Crosshair is drawn based on self.crosshair_3d mapped to each 2D view.
        # This seems more standard for neuroimaging viewers. Let's use this.

        # Trigger redraw of all slices using the updated current_positions and existing crosshair_3d
        self.update_all_slices()


    def update_all_slices(self):
        """Redraws all three slice views based on current positions and crosshair."""
        if self.brain_data is None or self.template_data is None:
            # Optionally clear views or show a placeholder
            for view in self.slice_views:
                 view.clear()
                 view.original_pixmap = None # Ensure pixmap is cleared too
            return

        try:
            # Get full 3D data arrays
            # Ensure correct data types, especially for atlas (int for indexing)
            brain_3d = self.brain_data.get_fdata(dtype=np.int32)
            template_3d = self.template_data.get_fdata(dtype=np.float32) # Use float for template intensity

            # Get current slice indices, ensuring they are within bounds
            max_x, max_y, max_z = brain_3d.shape
            # Note: Order is typically [x, y, z] for nibabel indexing
            # Sliders/positions might be [z, y, x] based on UI order. Double-check convention!
            # Assuming self.current_positions = [z, y, x] based on slider layout
            z_idx = min(max(self.current_positions[0], 0), max_z - 1)
            y_idx = min(max(self.current_positions[1], 0), max_y - 1)
            x_idx = min(max(self.current_positions[2], 0), max_x - 1)

            # Extract slices - Pay attention to indexing order and transposition for display
            # Nibabel shape: (X, Y, Z)
            # Axial (XY plane at Z=z_idx): brain_3d[:, :, z_idx] -> Shape (X, Y)
            # Coronal (XZ plane at Y=y_idx): brain_3d[:, y_idx, :] -> Shape (X, Z)
            # Sagittal (YZ plane at X=x_idx): brain_3d[x_idx, :, :] -> Shape (Y, Z)

            # Transpose as needed for conventional display orientation (e.g., Axial often shown with Anterior up)
            # View 0: Axial
            axial_slice = np.transpose(brain_3d[:, :, z_idx]) # Shape (Y, X) -> Display as (H, W)
            axial_template = np.transpose(template_3d[:, :, z_idx])
            # View 1: Coronal
            coronal_slice = np.transpose(brain_3d[:, y_idx, :]) # Shape (Z, X) -> Display as (H, W)
            coronal_template = np.transpose(template_3d[:, y_idx, :])
            # View 2: Sagittal
            sagittal_slice = np.transpose(brain_3d[x_idx, :, :]) # Shape (Z, Y) -> Display as (H, W)
            sagittal_template = np.transpose(template_3d[x_idx, :, :])

            # Determine highlight region (only for Practice mode after 3 errors)
            highlight_region = None
            if self.game_mode == "Practice" and self.consecutive_errors >= 3:
                highlight_region = self.current_target

            # Determine colormap (only if colored atlas is enabled)
            # Pass the actual colormap dict or None
            active_colormap = self.colormap if (self.use_colored_atlas and self.show_atlas) else None

            # Update each view
            self.slice_views[0].update_slice(axial_slice, axial_template, active_colormap, highlight_region, self.show_atlas)
            self.slice_views[1].update_slice(coronal_slice, coronal_template, active_colormap, highlight_region, self.show_atlas)
            self.slice_views[2].update_slice(sagittal_slice, sagittal_template, active_colormap, highlight_region, self.show_atlas)

            # Update crosshair position in all views based on the central 3D coordinate
            # Assuming self.crosshair_3d = (voxel_x, voxel_y, voxel_z)
            voxel_x, voxel_y, voxel_z = self.crosshair_3d
            # Ensure crosshair coords are within bounds too (although usually set by click/data limits)
            voxel_x = min(max(voxel_x, 0), max_x - 1)
            voxel_y = min(max(voxel_y, 0), max_y - 1)
            voxel_z = min(max(voxel_z, 0), max_z - 1)

            # The BrainSliceView.set_crosshair_3d method handles mapping the 3D coord to its 2D plane
            for view in self.slice_views:
                view.set_crosshair_3d(voxel_x, voxel_y, voxel_z)

        except IndexError as e:
             print(f"Error updating slices (IndexError): {e}. Check slice indices and data shapes.")
             # Maybe display an error message or clear views
             for view in self.slice_views:
                 view.clear()
                 view.original_pixmap = None
        except Exception as e:
             print(f"Unexpected error updating slices: {e}")
             # Handle other potential errors during slicing or data access


    def select_new_target(self):
        """Selects a new target region based on the game mode."""
        if not self.region_map:
             QMessageBox.critical(self, "Target Error", "Region map not loaded. Cannot select target.")
             self.end_game() # Cannot proceed
             return

        if self.game_mode == "Contre la Montre":
            if not self.remaining_regions:
                print("All regions found in Contre la Montre!")
                self.end_game() # Game finished successfully
                return
            self.current_target = self.remaining_regions.pop(0) # Get next from randomized list

        else: # Practice or Streak
            # Get list of all valid region IDs present in the atlas data
            # Check against self.region_map as well to ensure names exist
            unique_ids = np.unique(self.brain_data.get_fdata().astype(np.int32))
            valid_regions = [int(val) for val in unique_ids if val > 0 and val in self.region_map]

            if not valid_regions:
                QMessageBox.warning(self, "Target Error", "No valid regions found in the current atlas data.")
                self.end_game() # Cannot proceed
                return

            # Avoid picking the same target consecutively (simple approach)
            possible_targets = [r for r in valid_regions if r != self.current_target]
            if not possible_targets: possible_targets = valid_regions # Fallback if only one region

            if self.game_mode == "Streak" and self.streak_guessed_regions:
                 # Optional: Weight against recently guessed regions to encourage variety
                 # More complex: Implement weighted random choice
                 weights = [0.1 if region in self.streak_guessed_regions[-5:] else 1.0 for region in possible_targets] # Heavily weight against last 5
                 total_weight = sum(weights)
                 if total_weight > 0:
                     weights = [w / total_weight for w in weights]
                     try:
                         self.current_target = np.random.choice(possible_targets, p=weights)
                     except ValueError as e:
                          print(f"Warning: Error during weighted choice ({e}), using simple random choice.")
                          self.current_target = random.choice(possible_targets)

                 else: # Should not happen if possible_targets is not empty
                      self.current_target = random.choice(possible_targets)

            else: # Practice or start of Streak
                self.current_target = random.choice(possible_targets)

        # --- Update UI ---
        target_name = self.region_map.get(self.current_target, f"ID: {self.current_target}")
        self.target_label.setText(f"Find: {target_name}")
        self.consecutive_errors = 0 # Reset error counter for this target
        for view in self.slice_views:
             view.stop_blinking() # Ensure blinking stops when new target selected
        self.update_memo_content() # Update info panel (for Practice mode)
        self.selected_position = None # Require a new click for the new target
        self.guess_button.setEnabled(False) # Disable guess until next click
        self.guess_button.setText("Confirm Guess (Space)") # Reset text

        print(f"New target selected: {target_name} (ID: {self.current_target})")


    def handle_slice_click(self, x, y, plane_index):
        """Handles clicks on a BrainSliceView, updating the 3D crosshair and selected position."""
        if not self.game_running or not self.brain_data:
            return # Ignore clicks if game not running or no data

        brain_shape = self.brain_data.shape # (X, Y, Z)
        max_x, max_y, max_z = brain_shape

        # Map the 2D click (x, y in the slice's coordinate system) back to 3D voxel coordinates
        current_z, current_y, current_x = self.current_positions # Get current slice indices

        if plane_index == 0: # Axial View (Displayed as Y vs X)
            # Input x is image X, input y is image Y
            voxel_x = min(max(x, 0), max_x - 1)
            voxel_y = min(max(y, 0), max_y - 1)
            voxel_z = current_z # Z is fixed by the current axial slice index
        elif plane_index == 1: # Coronal View (Displayed as Z vs X)
            # Input x is image X, input y is image Z
            voxel_x = min(max(x, 0), max_x - 1)
            voxel_y = current_y # Y is fixed by the current coronal slice index
            voxel_z = min(max(y, 0), max_z - 1)
        elif plane_index == 2: # Sagittal View (Displayed as Z vs Y)
            # Input x is image Y, input y is image Z
            voxel_x = current_x # X is fixed by the current sagittal slice index
            voxel_y = min(max(x, 0), max_y - 1)
            voxel_z = min(max(y, 0), max_z - 1)
        else:
            return # Should not happen

        # --- Update the central 3D position ---
        self.selected_position = (voxel_x, voxel_y, voxel_z) # Store the clicked 3D coordinate
        self.crosshair_3d = (voxel_x, voxel_y, voxel_z) # Update the visual crosshair position

        # --- Optional: Snap sliders/views to the clicked coordinate ---
        # Comment this section out if you prefer sliders to remain independent
        self.current_positions = [voxel_z, voxel_y, voxel_x] # Update slice indices based on click
        # Block signals while setting slider values programmatically
        self.z_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.x_slider.blockSignals(True)
        self.z_slider.setValue(voxel_z)
        self.y_slider.setValue(voxel_y)
        self.x_slider.setValue(voxel_x)
        self.z_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.x_slider.blockSignals(False)
        # --- End of Optional Snapping ---

        # --- Redraw all slices to show the new crosshair position ---
        self.update_all_slices()

        # --- Enable the guess button ---
        self.guess_button.setEnabled(True)
        self.guess_button.setText(f"Guess: {self.region_map.get(self._get_region_at_pos(self.selected_position), '?')}") # Show clicked region name on button
        # self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px; font-weight: bold;") # Green style for enabled guess


    def _get_region_at_pos(self, position_3d):
        """Helper function to get the region ID at a specific 3D voxel coordinate."""
        if not position_3d or not self.brain_data:
            return 0 # Return 0 (background) if no position or data

        voxel_x, voxel_y, voxel_z = position_3d
        brain_3d_data = self.brain_data.get_fdata(dtype=np.int32)

        # Check bounds carefully
        if 0 <= voxel_x < brain_3d_data.shape[0] and \
           0 <= voxel_y < brain_3d_data.shape[1] and \
           0 <= voxel_z < brain_3d_data.shape[2]:
            try:
                return int(brain_3d_data[voxel_x, voxel_y, voxel_z])
            except IndexError:
                 print(f"IndexError getting region at {position_3d}")
                 return 0 # Should be caught by bounds check, but safety first
        else:
            # Click was outside the data volume (can happen with padding/display issues)
            print(f"Warning: Click position {position_3d} is outside data bounds {brain_3d_data.shape}")
            return 0 # Treat as background


    def validate_guess(self):
        """Checks if the selected position matches the current target region."""
        if not self.selected_position or not self.game_running or self.current_target is None:
             print("Validation skipped: No position selected, game not running, or no target.")
             return

        # Get the region ID at the clicked location
        clicked_region_id = self._get_region_at_pos(self.selected_position)

        target_name = self.region_map.get(self.current_target, f"ID {self.current_target}")
        # Get name, default to "Background/OOB" if ID is 0 or not in map
        clicked_name = self.region_map.get(clicked_region_id, "Background/OOB" if clicked_region_id == 0 else f"Unknown ID {clicked_region_id}")

        print(f"Validating guess: Target={target_name}({self.current_target}), Clicked={clicked_name}({clicked_region_id})")

        # --- Check if Correct ---
        if clicked_region_id == self.current_target:
            # --- CORRECT GUESS ---
            self.score += 1
            self.correct_guesses.append(target_name)
            self.consecutive_errors = 0 # Reset error counter
            for view in self.slice_views:
                view.stop_blinking() # Stop blinking if it was active

            # Update score display based on mode
            if self.game_mode == "Practice":
                self.score_label.setText(f"Correct: {self.score}")
            elif self.game_mode == "Streak":
                self.streak_guessed_regions.append(self.current_target) # Add to streak list
                self.score_label.setText(f"Streak: {self.score}")
            elif self.game_mode == "Contre la Montre":
                self.score_label.setText(f"Found: {self.score} / {len(self.all_regions)}")

            # Show feedback (optional, maybe less intrusive feedback later)
            # QMessageBox.information(self, "Correct!", f"Well done! You found the {target_name}!")
            # Use guess button for temporary feedback
            self.guess_button.setText("Correct!")
            self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px; font-weight: bold;") # Green feedback


            # Select next target or end game
            if self.game_mode == "Contre la Montre" and not self.remaining_regions:
                 # Short delay before ending to show "Correct!"
                 QTimer.singleShot(800, self.end_game)
                 # self.end_game()
            else:
                 # Short delay before selecting next target
                 QTimer.singleShot(800, self.select_new_target)
                 # self.select_new_target()

        else:
            # --- INCORRECT GUESS ---
            self.errors += 1
            self.consecutive_errors += 1
            self.incorrect_guesses.append((target_name, clicked_name)) # Log the error details
            self.error_label.setText(f"Errors: {self.errors}")

            # Give feedback
            feedback_message = f"Incorrect. That's the {clicked_name}."
            if self.game_mode != "Streak": # Don't reveal target in Streak
                 feedback_message += f"\nKeep looking for the {target_name}."

            # Use guess button for temporary feedback
            self.guess_button.setText("Incorrect!")
            self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #f44336; color: white; border-radius: 5px; font-weight: bold;") # Red feedback

            # Reset button state after a short delay
            QTimer.singleShot(1000, lambda: self.guess_button.setText(f"Guess: {clicked_name}")) # Show what was clicked
            QTimer.singleShot(1000, lambda: self.guess_button.setEnabled(True)) # Re-enable guess
            # QTimer.singleShot(1000, lambda: self.guess_button.setStyleSheet("...") ) # Reset style if needed


            # Handle mode-specific consequences
            if self.game_mode == "Practice":
                self.score_label.setText(f"Correct: {self.score}") # Score doesn't change
                if self.consecutive_errors >= 3:
                    feedback_message += "\nThe correct region is now blinking!"
                    for view in self.slice_views:
                        view.start_blinking() # Start blinking the target region
                # Show message box only after blinking starts or if no blinking
                QMessageBox.warning(self, "Incorrect Guess", feedback_message)


            elif self.game_mode == "Streak":
                # End the game immediately on the first error
                QMessageBox.warning(self, "Streak Over!", f"Incorrect. That was the {clicked_name}.\nYour streak ends at {self.score}.")
                self.end_game()

            elif self.game_mode == "Contre la Montre":
                # Score doesn't change, timer continues
                self.score_label.setText(f"Found: {self.score} / {len(self.all_regions)}")
                # Maybe less intrusive feedback than a message box?
                # For now, keep it for clarity.
                QMessageBox.warning(self, "Incorrect Guess", feedback_message)

            # Re-enable guess button if not ending game
            if self.game_running:
                 self.guess_button.setEnabled(True)
                 # Optionally reset button text after a delay
                 # QTimer.singleShot(1500, lambda: self.guess_button.setText("Confirm Guess (Space)"))

    def update_timer(self):
        """Callback for the game timer (Contre la Montre mode)."""
        if self.game_mode != "Contre la Montre" or not self.game_running or not self.start_time:
            self.game_timer.stop() # Stop timer if mode changed or game stopped
            return

        self.total_time = int(time.time() - self.start_time)
        minutes = self.total_time // 60
        seconds = self.total_time % 60
        self.timer_label.setText(f"Time: {minutes}'{seconds:02d}\"")

        # Optional: Add time limit condition
        # time_limit = 300 # e.g., 5 minutes
        # if self.total_time >= time_limit:
        #     QMessageBox.information(self, "Time's Up!", f"Time limit of {time_limit // 60} minutes reached.")
        #     self.end_game()


    def end_game(self):
        """Stops the game, calculates results, updates PRs, and shows summary."""
        if not self.game_running: return # Prevent double execution
        print("Ending game...")
        self.game_running = False
        self.game_timer.stop() # Stop timer if running
        for view in self.slice_views:
             view.stop_blinking() # Ensure blinking stops

        # --- Calculate Final Stats ---
        final_score = self.score
        final_errors = self.errors
        if (final_score + final_errors) > 0:
             accuracy = (final_score / (final_score + final_errors)) * 100.0
        else:
             accuracy = 100.0 if final_score > 0 else 0.0 # Handle division by zero / no attempts

        final_time = self.total_time # Only relevant for Contre la Montre
        final_streak = self.score # Only relevant for Streak mode

        # --- Update Personal Records (PR) ---
        atlas = self.current_atlas
        color_mode_key = "colored" if self.use_colored_atlas else "non_colored"
        new_pr_achieved = False
        pr_update_messages = []

        # Ensure the atlas and color mode structure exists
        if atlas not in self.pr_data: self.pr_data[atlas] = {"colored": self.load_pr()[atlas]["colored"], "non_colored": self.load_pr()[atlas]["non_colored"]} # Use default structure from load_pr
        if color_mode_key not in self.pr_data[atlas]: self.pr_data[atlas][color_mode_key] = self.load_pr()[atlas][color_mode_key]

        current_pr = self.pr_data[atlas][color_mode_key]

        # Update Best Accuracy (Ratio) - relevant for Practice & Contre la Montre
        if self.game_mode in ["Practice", "Contre la Montre"]:
             if accuracy > current_pr.get("best_ratio", 0.0):
                 current_pr["best_ratio"] = accuracy
                 new_pr_achieved = True
                 pr_update_messages.append(f"🏆 New Best Accuracy: {accuracy:.1f}%")

        # Update Best Time (only for Contre la Montre with 0 errors and all regions found)
        if self.game_mode == "Contre la Montre":
             # Check if all regions were found
             all_found = (final_score == len(self.all_regions))
             if all_found and final_errors == 0:
                  if final_time < current_pr.get("time", float("inf")):
                      current_pr["time"] = final_time
                      new_pr_achieved = True
                      minutes = final_time // 60
                      seconds = final_time % 60
                      pr_update_messages.append(f"⏱️ New Best Time: {minutes}'{seconds:02d}\"")
             elif all_found:
                 # Completed but with errors, maybe record errors for best time?
                 # current_pr["errors_at_best_time"] = final_errors # Optional
                 pass


        # Update Best Streak (only for Streak mode)
        if self.game_mode == "Streak":
             if final_streak > current_pr.get("best_streak", 0):
                 current_pr["best_streak"] = final_streak
                 new_pr_achieved = True
                 pr_update_messages.append(f"🔥 New Best Streak: {final_streak}")

        # Save PRs if any were updated
        if new_pr_achieved:
            self.save_pr()
            # self.update_pr_label() # Update labels on landing page (will be seen next time)
            print("New personal record(s) saved.")


        # --- Prepare Recap Message ---
        recap_title = "Game Over!"
        recap_message = f"Mode: {self.game_mode}\nAtlas: {atlas} ({'Colored' if self.use_colored_atlas else 'No Colors'})\n\n"

        if self.game_mode == "Contre la Montre":
             recap_title = "Finished!" if all_found else "Time Ran Out / Stopped"
             recap_message += f"Regions Found: {final_score} / {len(self.all_regions)}\n"
             recap_message += f"Time Taken: {final_time // 60}'{final_time % 60:02d}\"\n"
             recap_message += f"Errors: {final_errors}\n"
             recap_message += f"Accuracy: {accuracy:.1f}%\n"

        elif self.game_mode == "Streak":
             recap_title = "Streak Ended!"
             recap_message += f"Final Streak: {final_score}\n"
             # Error count is always 1 if ended by mistake, 0 if ended manually? No, errors field tracks all errors.
             # recap_message += f"Errors Made: {final_errors}\n" # Usually 1, unless stopped manually

        else: # Practice
             recap_title = "Practice Session Ended"
             recap_message += f"Correct Guesses: {final_score}\n"
             recap_message += f"Errors: {final_errors}\n"
             recap_message += f"Accuracy: {accuracy:.1f}%\n"


        # Add PR messages
        if pr_update_messages:
             recap_message += "\n--- New Records! ---\n" + "\n".join(pr_update_messages) + "\n"

        # Add details of errors (optional)
        if final_errors > 0 and self.incorrect_guesses:
            recap_message += "\n--- Errors Made ---\n"
            # Limit displayed errors if too many?
            display_errors = self.incorrect_guesses[:10] # Show max 10 errors
            for target, clicked in display_errors:
                recap_message += f"- Target: {target}, Clicked: {clicked}\n"
            if len(self.incorrect_guesses) > 10:
                 recap_message += f"... and {len(self.incorrect_guesses) - 10} more errors.\n"

        # --- Show Summary Dialog ---
        QMessageBox.information(self, recap_title, recap_message)

        # --- Reset UI and switch back to Landing Page ---
        self.reset_game_ui() # Reset buttons, scores, etc.
        self.update_pr_label() # Update PR display now that game is over
        self.stacked_widget.setCurrentWidget(self.landing_widget) # Go back to menu


    def show_help(self):
        """Displays instructions based on the current game mode."""
        help_title = f"How to Play: {self.game_mode} Mode"
        help_text = f"Atlas: {self.current_atlas} ({'Colored' if self.use_colored_atlas else 'No Colors'})\n"
        help_text += f"Template: {self.current_template_name}\n\n"
        help_text += "General Controls:\n"
        help_text += "- Click/Drag in views: Move crosshair to select a point.\n"
        help_text += "- Mouse Wheel: Scroll through slices in the hovered view.\n"
        help_text += "- Ctrl + Mouse Wheel: Zoom in/out in the hovered view.\n"
        help_text += "- Sliders: Navigate through slices for each plane.\n"
        help_text += "- Spacebar: Confirm your guess (when enabled).\n"
        help_text += "- Show Atlas Regions: Toggle visibility of colored overlay.\n"
        help_text += f"- {self.template_switch_button.text()}: Switch background template.\n\n"


        if self.game_mode == "Practice":
            help_text += "Practice Mode:\n"
            help_text += "1. A brain region name will appear as the 'Target'.\n"
            help_text += "2. Use the views and crosshair to locate that region.\n"
            help_text += "3. Click 'Confirm Guess' or press Spacebar when you think the crosshair is on the target.\n"
            help_text += "4. The panel on the right shows information about the target region.\n"
            help_text += "5. If you make 3 mistakes on the same target, the correct region will blink.\n"
            help_text += "6. There is no time limit. Click 'Menu' to end the session."

        elif self.game_mode == "Contre la Montre":
            help_text += "Contre la Montre Mode:\n"
            help_text += "1. Find all unique regions in the selected atlas as quickly as possible.\n"
            help_text += "2. A target region is given. Locate it and confirm your guess.\n"
            help_text += "3. If correct, the next target appears. If incorrect, you get an error, but the target remains the same.\n"
            help_text += "4. The timer runs until all regions are found or you click 'Menu'.\n"
            help_text += "5. Aim for the fastest time with the fewest errors!"

        elif self.game_mode == "Streak":
             help_text += "Streak Mode:\n"
             help_text += "1. Find as many regions as possible consecutively without making a mistake.\n"
             help_text += "2. A target region is given. Locate it and confirm your guess.\n"
             help_text += "3. If correct, your streak increases, and a new target appears.\n"
             help_text += "4. The game ends immediately on your first incorrect guess.\n"
             help_text += "5. Aim for the longest streak!"

        QMessageBox.information(self, help_title, help_text)


    def show_menu(self):
        """Returns to the landing page, stopping the current game if running."""
        if self.game_running:
            reply = QMessageBox.question(self, "Return to Menu?",
                                         "Are you sure you want to end the current game and return to the main menu?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                 print("Returning to menu, ending game.")
                 self.end_game() # This will show summary and reset UI before switching
                 # The end_game function already switches back to landing_widget
            else:
                 print("Returning to menu cancelled.")
                 return # Don't do anything if user cancels
        else:
             # If game wasn't running, just switch view
             print("Returning to menu (no game running).")
             self.reset_game_ui() # Ensure UI is clean
             self.update_pr_label() # Make sure PRs are up-to-date
             self.stacked_widget.setCurrentWidget(self.landing_widget)


    def closeEvent(self, event):
        """Handle window close event."""
        # Optional: Add confirmation dialog if a game is running
        if self.game_running:
             reply = QMessageBox.question(self, "Quit Game?",
                                         "A game is currently in progress. Are you sure you want to quit?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.No:
                 event.ignore() # Don't close
                 return

        # Optional: Save any unsaved data (like PRs, though they are saved on update)
        # self.save_pr()

        print("Closing NeuroGuessr.")
        event.accept() # Proceed with closing


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Fusion style often works well cross-platform

    # --- Dark Theme Palette ---
    dark_palette = QPalette()
    # Base colors
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))       # Main window background
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255)) # Default text color
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))         # Background for text inputs, lists etc.
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))# Alternate row color (if used)
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))  # Tooltip background
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))# Tooltip text
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))       # Text color in input widgets
    # Button colors
    dark_palette.setColor(QPalette.Button, QColor(66, 66, 66))       # Button background
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255)) # Text on buttons
    # Highlight colors
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))  # Selection highlight (e.g., in lists)
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))# Text color when highlighted
    # Disabled colors
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }") # Style tooltips


    # --- Font Loading (Optional) ---
    # font_id = QFontDatabase.addApplicationFont(get_resource_path("fonts/YourCustomFont.ttf")) # Add custom font
    # if font_id != -1:
    #     font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
    #     app.setFont(QFont(font_family, 10)) # Set default app font
    # else:
    #     print("Warning: Failed to load custom font.")

    # --- Initialize and Run ---
    game = NeuroGuessrGame()
    game.show()
    sys.exit(app.exec_())