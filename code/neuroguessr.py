import os
import sys
import time
import random
import numpy as np
import pandas as pd
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QGridLayout, QSplitter, QFrame,
                             QSlider, QProgressBar, QLCDNumber, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont, QPalette, QImage


# update zoom
# update slice selon le click axial/sagittal etcpyt
# mettre 15 sec par region plutot que temsp global.

class BrainSliceView(QLabel):
    """Widget to display a single brain slice with click and zoom functionality."""
    slice_clicked = pyqtSignal(int, int, int)  # x, y, plane_index

    def __init__(self, plane_index, parent=None):
        super().__init__(parent)
        self.plane_index = plane_index
        self.crosshair_pos = (0, 0)  # Position 2D dans la coupe
        self.zoom_factor = 1.5
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.slice_data = None
        self.plane_names = ["Axial", "Coronal", "Sagittal"]
        self.original_pixmap = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        
        self.title = QLabel(self.plane_names[plane_index])
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 0);")
        self.title.setFont(QFont("Arial", 12, QFont.Bold))

    def set_crosshair_3d(self, voxel_x, voxel_y, voxel_z):
        """Mettre à jour la position du crosshair en fonction des coordonnées 3D."""
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
            
        # Obtenir le delta de la molette
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.2  # Zoom avant
        elif delta < 0:
            self.zoom_factor /= 1.2  # Zoom arrière
            
        # Limiter le zoom entre 0.5x et 5x
        self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))
        
        # Mettre à jour l'affichage
        self.update()
        event.accept()  # Accepter l'événement
        
    def update_slice(self, slice_data, colormap=None):
        """Update the displayed brain slice."""
        self.slice_data = slice_data
        
        if slice_data is None:
            self.clear()
            return
            
        # Normaliser les données pour l'affichage
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
            
        # Stocker l'image originale
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update()
        
    def paintEvent(self, event):
        """Override paint event to draw crosshairs and handle zoom."""
        super().paintEvent(event)
        
        if not self.original_pixmap:
            return
            
        painter = QPainter(self)
        
        # Calculer les dimensions de l'image zoomée
        img_width = int(self.original_pixmap.width() * self.zoom_factor)
        img_height = int(self.original_pixmap.height() * self.zoom_factor)
        
        # Centrer l'image
        label_width = self.width()
        label_height = self.height()
        x_offset = (label_width - img_width) // 2
        y_offset = (label_height - img_height) // 2
        
        # Appliquer le zoom
        scaled_pixmap = self.original_pixmap.scaled(img_width, img_height, 
                                                  Qt.KeepAspectRatio, 
                                                  Qt.SmoothTransformation)
        
        # Dessiner l'image
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # Dessiner les lignes de visée
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        
        x, y = self.crosshair_pos
        scaled_x = int(x * self.zoom_factor) + x_offset
        scaled_y = int(y * self.zoom_factor) + y_offset
        
        painter.drawLine(scaled_x, y_offset, scaled_x, y_offset + img_height)
        painter.drawLine(x_offset, scaled_y, x_offset + img_width, scaled_y)
        
        # Dessiner le titre
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(0, 20, label_width, 20, Qt.AlignCenter, self.plane_names[self.plane_index])
        
        painter.end()
        
    def mousePressEvent(self, event):
        """Handle mouse click events with zoom adjustment."""
        if not self.original_pixmap:
            return
            
        # Calculer la position du clic dans l'image originale
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
        
        if (0 <= orig_x < self.original_pixmap.width() and 
            0 <= orig_y < self.original_pixmap.height()):
            # Émettre le signal sans modifier crosshair_pos ici
            self.slice_clicked.emit(orig_x, orig_y, self.plane_index)


class NeuroGuessrGame(QMainWindow):
    """Main window for the NeuroGuessr game."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGuessr - Test Your Brain Region Knowledge!")
        self.resize(1200, 800)
        
        # Game state variables
        self.score = 0
        self.errors = 0
        self.current_target = None
        self.game_running = False
        self.time_remaining = 50
        self.regions_to_find = 10
        self.regions_found = 0
        self.brain_data = None
        self.region_map = None
        self.colormap = {}
        self.current_slices = [None, None, None]
        self.current_positions = [0, 0, 0]  # z, y, x positions
        self.selected_position = None
        self.crosshair_3d = (0, 0, 0)  # Coordonnée 3D commune pour les crosshairs
        
        self.setup_ui()
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_timer)
        self.load_data()

    def start_game(self):
        """Start or restart the game."""
        self.score = 0
        self.errors = 0
        self.regions_found = 0
        self.time_remaining = 50
        self.game_running = True
        
        self.score_label.setText(f"Score: {self.score}")
        self.error_label.setText(f"Errors: {self.errors}")
        self.timer_display.display(self.time_remaining)
        self.start_button.setText("Restart Game")
        self.guess_button.setEnabled(False)
        
        self.select_new_target()
        self.game_timer.start(1000)
        
    def setup_ui(self):
        """Set up the user interface."""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Game status area
        status_layout = QHBoxLayout()
        
        # Target region display
        self.target_label = QLabel("Target: Not Started")
        self.target_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.target_label.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 5px;")
        self.target_label.setAlignment(Qt.AlignCenter)
        
        # Timer display
        self.timer_display = QLCDNumber()
        self.timer_display.setDigitCount(3)
        self.timer_display.display(60)
        self.timer_display.setStyleSheet("background-color: #333;")
        
        # Score display
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
        
        # Brain slice views area
        views_layout = QHBoxLayout()
        
        # Create the three slice view widgets
        self.slice_views = []
        for i in range(3):
            view = BrainSliceView(i)
            view.slice_clicked.connect(self.handle_slice_click)
            self.slice_views.append(view)
            views_layout.addWidget(view)
        
        main_layout.addLayout(views_layout, 1)
        
        # Slider controls
        slider_layout = QHBoxLayout()
        
        # Axial slice slider (Z)
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
        
        # Coronal slice slider (Y)
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
        
        # Sagittal slice slider (X)
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
        
        # Game control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Game")
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px;")
        
        self.guess_button = QPushButton("Confirm Guess")
        self.guess_button.clicked.connect(self.validate_guess)
        self.guess_button.setEnabled(False)  # Disabled until a region is selected
        self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
        
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.guess_button)
        button_layout.addWidget(self.help_button)
        
        main_layout.addLayout(button_layout)
        
        self.setCentralWidget(main_widget)
        
    def load_data(self):
        """Load brain data and region mapping from files."""
        try:
            # For demonstration, we'll create dummy data if files aren't found
            data_loaded = False
            
            # Try to load the actual NIFTI file and region mapping
            if os.path.exists("/Users/francoisramon/Desktop/These/neuroguessr/data/aparc_DKT_stride.nii.gz") and os.path.exists("/Users/francoisramon/Desktop/These/neuroguessr/data/fs_a2009s.txt"):
                try:
                    self.brain_data = nib.load("/Users/francoisramon/Desktop/These/neuroguessr/data/aparc_DKT_stride.nii.gz")
                    region_df = pd.read_csv("/Users/francoisramon/Desktop/These/neuroguessr/data/fs_a2009s.txt", sep="\s+", comment="#", header=None, 
                                           names=["Index", "RegionName", "R", "G", "B", "A"])

                    print(region_df)
                    
                    # Create region map and colormap
                    self.region_map = {row["Index"]: row["RegionName"] for _, row in region_df.iterrows()}
                    
                    # Create colormap for visualization
                    for _, row in region_df.iterrows():
                        self.colormap[row["Index"]] = (row["R"], row["G"], row["B"])
                    
                    data_loaded = True
                    
                    # Update sliders based on actual dimensions
                    self.z_slider.setMaximum(self.brain_data.shape[2] - 1)
                    self.y_slider.setMaximum(self.brain_data.shape[1] - 1)
                    self.x_slider.setMaximum(self.brain_data.shape[0] - 1)
                    
                    # Update with middle slices
                    self.z_slider.setValue(self.brain_data.shape[2] // 2)
                    self.y_slider.setValue(self.brain_data.shape[1] // 2)
                    self.x_slider.setValue(self.brain_data.shape[0] // 2)
                    
                    # Initial update of slice views
                    self.update_all_slices()
                    
                except Exception as e:
                    print(f"Error loading real data: {e}")
            
            if not data_loaded:
                # Create dummy data for demonstration
                print("Creating dummy brain data for demonstration")
                dummy_shape = (128, 128, 128)
                dummy_data = np.zeros(dummy_shape, dtype=np.int16)
                
                # Create some dummy regions
                regions = {
                    1: "Left Cerebral Cortex",
                    2: "Right Cerebral Cortex",
                    3: "Left Hippocampus",
                    4: "Right Hippocampus",
                    5: "Left Thalamus",
                    6: "Right Thalamus",
                    7: "Left Amygdala",
                    8: "Right Amygdala",
                    9: "Left Caudate",
                    10: "Right Caudate"
                }
                
                # Create dummy colormap
                for i in range(1, 11):
                    self.colormap[i] = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)
                    )
                
                # Add some random shapes for the regions
                for i in range(1, 11):
                    center = np.array([
                        random.randint(30, 98),
                        random.randint(30, 98),
                        random.randint(30, 98)
                    ])
                    size = random.randint(10, 20)
                    # Which side: left (even) or right (odd)
                    if i % 2 == 0:  # right
                        center[0] = random.randint(70, 98)
                    else:  # left
                        center[0] = random.randint(30, 58)
                        
                    # Create an ellipsoid for the region
                    for x in range(max(0, center[0]-size), min(dummy_shape[0], center[0]+size)):
                        for y in range(max(0, center[1]-size), min(dummy_shape[1], center[1]+size)):
                            for z in range(max(0, center[2]-size), min(dummy_shape[2], center[2]+size)):
                                # Ellipsoid equation
                                if ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2) < size**2:
                                    dummy_data[x, y, z] = i
                
                # Create affine matrix for the dummy data
                affine = np.eye(4)
                self.brain_data = nib.Nifti1Image(dummy_data, affine)
                self.region_map = regions
                
                # Update sliders
                self.z_slider.setMaximum(dummy_shape[2] - 1)
                self.y_slider.setMaximum(dummy_shape[1] - 1)
                self.x_slider.setMaximum(dummy_shape[0] - 1)
                
                # Set sliders to middle positions
                self.z_slider.setValue(dummy_shape[2] // 2)
                self.y_slider.setValue(dummy_shape[1] // 2)
                self.x_slider.setValue(dummy_shape[0] // 2)
                
                # Initial update of slice views
                self.update_all_slices()
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", 
                                f"Failed to load brain data: {str(e)}\n\n"
                                "The application will use dummy data for demonstration.")
            
    def update_slice_position(self, plane_index, value):
        """Update a specific slice position (slider callback)."""
        self.current_positions[plane_index] = value
        
        # Mettre à jour crosshair_3d pour refléter la nouvelle position
        if plane_index == 0:  # Axial (z)
            self.crosshair_3d = (self.crosshair_3d[0], self.crosshair_3d[1], value)
        elif plane_index == 1:  # Coronal (y)
            self.crosshair_3d = (self.crosshair_3d[0], value, self.crosshair_3d[2])
        elif plane_index == 2:  # Sagittal (x)
            self.crosshair_3d = (value, self.crosshair_3d[1], self.crosshair_3d[2])
        
        self.update_all_slices()
        
    def update_all_slices(self):
        """Update all three slice views based on current 3D position."""
        if self.brain_data is None:
            return
            
        brain_3d = self.brain_data.get_fdata().astype(np.int32)
        z, y, x = self.current_positions
        
        # S'assurer que les indices sont dans les limites
        brain_shape = self.brain_data.shape
        x = min(max(x, 0), brain_shape[0] - 1)
        y = min(max(y, 0), brain_shape[1] - 1)
        z = min(max(z, 0), brain_shape[2] - 1)
        
        # Extraire les coupes
        axial_slice = brain_3d[:, :, z].T  # xy plane
        coronal_slice = brain_3d[:, y, :].T  # xz plane
        sagittal_slice = brain_3d[x, :, :].T  # yz plane
        
        # Mettre à jour les vues
        self.slice_views[0].update_slice(axial_slice, self.colormap)
        self.slice_views[1].update_slice(coronal_slice, self.colormap)
        self.slice_views[2].update_slice(sagittal_slice, self.colormap)
        
        # Mettre à jour les crosshairs
        voxel_x, voxel_y, voxel_z = self.crosshair_3d
        for view in self.slice_views:
            view.set_crosshair_3d(voxel_x, voxel_y, voxel_z)
        
    def select_new_target(self):
        """Select a new target region for the player to find."""
        # Only consider regions that actually exist in the data (non-zero voxels)
        valid_regions = []
        brain_data = self.brain_data.get_fdata().astype(np.int32)
        
        unique_values = np.unique(brain_data)
        for value in unique_values:
            if value > 0 and value in self.region_map:
                valid_regions.append(int(value))
        
        if not valid_regions:
            QMessageBox.warning(self, "Error", "No valid regions found in the data.")
            return
            
        # Select a random region
        region_id = random.choice(valid_regions)
        print(region_id)
        region_name = self.region_map[region_id]
        print(region_name)
        
        self.current_target = region_id
        self.target_label.setText(f"Find: {region_name}")
        
    def handle_slice_click(self, x, y, plane_index):
        """Handle click on a brain slice and update all views."""
        if not self.game_running or not self.brain_data:
            return
            
        # Convertir les coordonnées 2D du clic en coordonnées 3D
        brain_shape = self.brain_data.shape  # (x, y, z)
        if plane_index == 0:  # Axial (xy plane)
            voxel_x = min(max(x, 0), brain_shape[0] - 1)
            voxel_y = min(max(y, 0), brain_shape[1] - 1)
            voxel_z = self.current_positions[0]  # z reste inchangé pour l'instant
        elif plane_index == 1:  # Coronal (xz plane)
            voxel_x = min(max(x, 0), brain_shape[0] - 1)
            voxel_y = self.current_positions[1]  # y reste inchangé
            voxel_z = min(max(y, 0), brain_shape[2] - 1)
        else:  # Sagittal (yz plane)
            voxel_x = self.current_positions[2]  # x reste inchangé
            voxel_y = min(max(x, 0), brain_shape[1] - 1)
            voxel_z = min(max(y, 0), brain_shape[2] - 1)
        
        # Mettre à jour les positions des coupes pour centrer sur le point 3D
        self.current_positions = [voxel_z, voxel_y, voxel_x]  # [z, y, x]
        
        # Mettre à jour les sliders pour refléter les nouvelles positions
        self.z_slider.setValue(voxel_z)
        self.y_slider.setValue(voxel_y)
        self.x_slider.setValue(voxel_x)
        
        # Stocker la coordonnée 3D pour la validation
        self.selected_position = (voxel_x, voxel_y, voxel_z)
        self.crosshair_3d = (voxel_x, voxel_y, voxel_z)
        
        # Mettre à jour les crosshairs dans toutes les vues
        for view in self.slice_views:
            view.set_crosshair_3d(voxel_x, voxel_y, voxel_z)
        
        # Mettre à jour toutes les coupes
        self.update_all_slices()
        
        # Activer le bouton de guess
        self.guess_button.setEnabled(True)
        self.guess_button.setText("Confirm Guess")
        self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; font-weight: bold;")

    def validate_guess(self):
        """Validate the current guess when the guess button is clicked."""
        if not hasattr(self, 'selected_position') or not self.game_running:
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
                self.time_remaining = 20  # Réinitialiser à 20 secondes
                self.timer_display.display(self.time_remaining)
                self.select_new_target()
        else:
            self.errors += 1
            self.error_label.setText(f"Errors: {self.errors}")
            clicked_name = self.region_map.get(clicked_region, "Background/Unknown")
            target_name = self.region_map.get(self.current_target, "Unknown")
            QMessageBox.warning(self, "Incorrect", 
                              f"That's the {clicked_name}.\nYou need to find the {target_name}.")
        
        self.guess_button.setEnabled(False)
        self.guess_button.setText("Confirm Guess")
        self.guess_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50;")
        
    def update_timer(self):
        """Update the game timer."""
        self.time_remaining -= 1
        self.timer_display.display(self.time_remaining)
        
        if self.time_remaining <= 0:
            self.errors += 1
            self.error_label.setText(f"Errors: {self.errors}")
            QMessageBox.warning(self, "Time's Up!", 
                              f"Time ran out for this region!")
            self.regions_found += 1
            
            if self.regions_found >= self.regions_to_find:
                self.end_game()
            else:
                self.time_remaining = 20
                self.timer_display.display(self.time_remaining)
                self.select_new_target()
                
    def end_game(self):
        """End the current game."""
        self.game_running = False
        self.game_timer.stop()
        
        QMessageBox.information(self, "Game Over", 
                              f"Game finished!\nFinal Score: {self.score}/{self.regions_to_find}\nErrors: {self.errors}")
        
        self.start_button.setText("Start New Game")
        self.target_label.setText("Game Over")
        
    def show_help(self):
        """Show help information."""
        QMessageBox.information(self, "How to Play NeuroGuessr", 
                             "1. Click 'Start Game' to begin\n"
                             "2. You'll be given a brain region to find\n"
                             "3. Use the sliders to navigate through the brain\n"
                             "4. Click on the region you think matches the target\n"
                             "5. Click the 'Confirm Guess' button to validate your selection\n"
                             "6. Score points for correct guesses\n"
                             "7. Try to get as many correct as possible before time runs out!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look across platforms
    
    # Set dark theme
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