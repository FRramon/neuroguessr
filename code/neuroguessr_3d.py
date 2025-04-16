import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
import nibabel as nib
from vispy import scene
from vispy.scene.visuals import Mesh, Markers
from skimage.measure import marching_cubes

class NeuroGuessrGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGuessr with 3D View")
        self.brain_data = None
        self.colormap = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # Example colormap
        self.crosshair_3d = [128, 128, 128]  # Initial crosshair position
        self.setup_ui()

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Grid layout for views
        views_layout = QGridLayout()
        views_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder labels for 2D slice views (replace with actual slice widgets if available)
        self.slice_views = [
            QLabel("Axial View"),  # Top-left (0, 0)
            QLabel("Coronal View"),  # Top-right (0, 1)
            QLabel("Sagittal View")  # Bottom-left (1, 0)
        ]
        for view in self.slice_views:
            view.setAlignment(Qt.AlignCenter)
            view.setStyleSheet("background-color: #333; color: white;")

        # Add 2D slice views to grid
        views_layout.addWidget(self.slice_views[0], 0, 0)  # Axial
        views_layout.addWidget(self.slice_views[1], 0, 1)  # Coronal
        views_layout.addWidget(self.slice_views[2], 1, 0)  # Sagittal

        # Set up VisPy canvas for 3D view
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera()

        # Add VisPy canvas to grid (Bottom-right: 1, 1)
        views_layout.addWidget(self.canvas.native, 1, 1)

        # Make views equal in size
        views_layout.setColumnStretch(0, 1)
        views_layout.setColumnStretch(1, 1)
        views_layout.setRowStretch(0, 1)
        views_layout.setRowStretch(1, 1)

        main_layout.addLayout(views_layout)

        # Load dummy or real data
        self.load_data()

    def load_data(self, file_path=None):
        if file_path:
            # Load real NIfTI data
            img = nib.load(file_path)
            self.brain_data = img.get_fdata().astype(np.int32)
        else:
            # Generate dummy data (256x256x256 with some regions)
            self.brain_data = np.zeros((256, 256, 256), dtype=np.int32)
            self.brain_data[50:200, 50:200, 50:200] = 1  # Dummy brain region

        brain_shape = self.brain_data.shape

        # Extract brain surface for 3D view
        mask = self.brain_data > 0
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
        self.brain_mesh = Mesh(vertices=verts, faces=faces, color=(0.7, 0.7, 0.7))
        self.view.add(self.brain_mesh)

        # Add crosshair marker
        self.crosshair_marker = Markers()
        self.view.add(self.crosshair_marker)

        # Set camera to center the brain
        center = np.array(brain_shape) / 2
        distance = max(brain_shape) * 1.5
        self.view.camera.center = tuple(center)
        self.view.camera.distance = distance

        # Update initial 3D view
        self.update_3d_view()

        # Update 2D slices (placeholder implementation)
        self.update_all_slices()

    def update_all_slices(self):
        # Placeholder: Update 2D slice displays based on self.crosshair_3d
        axial_pos, coronal_pos, sagittal_pos = self.crosshair_3d
        self.slice_views[0].setText(f"Axial (z={axial_pos})")
        self.slice_views[1].setText(f"Coronal (y={coronal_pos})")
        self.slice_views[2].setText(f"Sagittal (x={sagittal_pos})")
        self.update_3d_view()

    def update_3d_view(self):
        if self.brain_data is None:
            return
        pos = np.array([self.crosshair_3d])
        self.crosshair_marker.set_data(pos, face_color='red', size=10)

    def handle_slice_click(self, slice_type, x, y):
        # Example: Update crosshair based on click (simplified)
        if slice_type == 'axial':
            self.crosshair_3d[0] = x
            self.crosshair_3d[1] = y
        elif slice_type == 'coronal':
            self.crosshair_3d[0] = x
            self.crosshair_3d[2] = y
        elif slice_type == 'sagittal':
            self.crosshair_3d[1] = x
            self.crosshair_3d[2] = y
        self.update_all_slices()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = NeuroGuessrGame()
    game.show()
    sys.exit(app.exec_())