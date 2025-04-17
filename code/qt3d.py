from PyQt5 import Qt3DCore
from PyQt5 import Qt3DExtras
from PyQt5 import Qt3DRender
from PyQt5.QtGui import QVector3D, QColor
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QButtonGroup
import os
import sys
import numpy as np

class MeshViewer(QMainWindow):
    def __init__(self, obj_file_path):
        super().__init__()
        self.setWindowTitle("Qt3D Mesh Viewer")
        self.obj_file_path = obj_file_path
        self.rotation_x = 0
        self.rotation_y = 0
        self.translation = QVector3D(0, 0, 0)
        self.last_mouse_pos = None
        self.camera_distance = 200.0  # Increased for further zoom out
        self.interaction_mode = "rotate"
        self.init_ui()

    def compute_mesh_centroid(self):
        """Parse .obj file to compute the centroid and bounding box of the mesh."""
        vertices = []
        try:
            with open(self.obj_file_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, z = map(float, parts[1:4])
                            vertices.append([x, y, z])
        except Exception as e:
            print(f"Error reading .obj file: {e}")
            return QVector3D(0, 0, 0)

        if not vertices:
            print("No vertices found in .obj file")
            return QVector3D(0, 0, 0)

        vertices = np.array(vertices)
        centroid = np.mean(vertices, axis=0)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        print(f"Centroid: {centroid}")
        print(f"Bounding box: min={min_coords}, max={max_coords}")
        return QVector3D(float(centroid[0]), float(centroid[1]), float(centroid[2]))

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Button layout
        button_layout = QHBoxLayout()
        self.move_button = QPushButton("Move")
        self.rotate_button = QPushButton("Rotate")
        button_layout.addWidget(self.move_button)
        button_layout.addWidget(self.rotate_button)
        main_layout.addLayout(button_layout)

        # Button group for exclusive selection
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.move_button, 1)
        self.button_group.addButton(self.rotate_button, 2)
        self.move_button.setCheckable(True)
        self.rotate_button.setCheckable(True)
        self.rotate_button.setChecked(True)
        self.button_group.buttonClicked.connect(self.set_interaction_mode)

        # Create 3D view
        self.view = Qt3DExtras.Qt3DWindow()
        self.view.defaultFrameGraph().setClearColor(QColor(50, 50, 50))
        container = QWidget.createWindowContainer(self.view)
        main_layout.addWidget(container)

        # Create scene
        self.root_entity = Qt3DCore.QEntity()
        self.view.setRootEntity(self.root_entity)

        # Compute centroid
        self.centroid = self.compute_mesh_centroid()

        # Load mesh
        self.mesh = Qt3DRender.QMesh()
        self.mesh.setSource(QUrl.fromLocalFile(os.path.abspath(self.obj_file_path)))

        # Create transform
        self.transform = Qt3DCore.QTransform()
        self.transform.setScale(0.1)  # Adjusted for MRI meshes
        self.transform.setTranslation(-self.centroid + self.translation)
        # Optional manual translation (uncomment and adjust if needed)
        # self.transform.setTranslation(QVector3D(-x, -y, -z))

        # Create material (fully opaque)
        self.material = Qt3DExtras.QDiffuseSpecularMaterial()
        self.material.setAmbient(QColor(200, 200, 200, 255))
        self.material.setDiffuse(QColor(150, 150, 150, 255))
        self.material.setSpecular(QColor(255, 255, 255, 255))
        self.material.setShininess(50.0)
        self.material.setAlphaBlendingEnabled(False)

        # Create mesh entity
        self.mesh_entity = Qt3DCore.QEntity(self.root_entity)
        self.mesh_entity.addComponent(self.mesh)
        self.mesh_entity.addComponent(self.transform)
        self.mesh_entity.addComponent(self.material)

        # Add axes at centroid
        axis_length = 1.0  # Length of each axis
        axis_radius = 0.05  # Thickness of axes
        # X-axis (red)
        x_axis_mesh = Qt3DExtras.QCylinderMesh()
        x_axis_mesh.setRadius(axis_radius)
        x_axis_mesh.setLength(axis_length)
        x_axis_transform = Qt3DCore.QTransform()
        x_axis_transform.setTranslation(self.centroid + QVector3D(axis_length / 2, 0, 0))
        x_axis_transform.setRotationZ(90)  # Align with X-axis
        x_axis_material = Qt3DExtras.QDiffuseSpecularMaterial()
        x_axis_material.setAmbient(QColor(255, 0, 0, 255))  # Red
        x_axis_entity = Qt3DCore.QEntity(self.root_entity)
        x_axis_entity.addComponent(x_axis_mesh)
        x_axis_entity.addComponent(x_axis_transform)
        x_axis_entity.addComponent(x_axis_material)

        # Y-axis (green)
        y_axis_mesh = Qt3DExtras.QCylinderMesh()
        y_axis_mesh.setRadius(axis_radius)
        y_axis_mesh.setLength(axis_length)
        y_axis_transform = Qt3DCore.QTransform()
        y_axis_transform.setTranslation(self.centroid + QVector3D(0, axis_length / 2, 0))
        y_axis_material = Qt3DExtras.QDiffuseSpecularMaterial()
        y_axis_material.setAmbient(QColor(0, 255, 0, 255))  # Green
        y_axis_entity = Qt3DCore.QEntity(self.root_entity)
        y_axis_entity.addComponent(y_axis_mesh)
        y_axis_entity.addComponent(y_axis_transform)
        y_axis_entity.addComponent(y_axis_material)

        # Z-axis (blue)
        z_axis_mesh = Qt3DExtras.QCylinderMesh()
        z_axis_mesh.setRadius(axis_radius)
        z_axis_mesh.setLength(axis_length)
        z_axis_transform = Qt3DCore.QTransform()
        z_axis_transform.setTranslation(self.centroid + QVector3D(0, 0, axis_length / 2))
        z_axis_transform.setRotationX(90)  # Align with Z-axis
        z_axis_material = Qt3DExtras.QDiffuseSpecularMaterial()
        z_axis_material.setAmbient(QColor(0, 0, 255, 255))  # Blue
        z_axis_entity = Qt3DCore.QEntity(self.root_entity)
        z_axis_entity.addComponent(z_axis_mesh)
        z_axis_entity.addComponent(z_axis_transform)
        z_axis_entity.addComponent(z_axis_material)

        # Add directional light
        self.light_entity = Qt3DCore.QEntity(self.root_entity)
        self.light = Qt3DRender.QDirectionalLight()
        self.light.setWorldDirection(QVector3D(0, 0, -1))
        self.light.setIntensity(0.8)
        self.light.setColor(QColor(255, 255, 255))
        self.light_entity.addComponent(self.light)

        # Set up camera
        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 10000.0)
        self.camera.setPosition(QVector3D(0, 0, self.camera_distance))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

        # Camera controller
        self.cam_controller = Qt3DExtras.QOrbitCameraController(self.root_entity)
        self.cam_controller.setCamera(self.camera)
        self.cam_controller.setLinearSpeed(0)
        self.cam_controller.setLookSpeed(0)

        # Enable mouse events
        self.view.installEventFilter(self)

    def set_interaction_mode(self, button):
        """Set the interaction mode based on button clicked."""
        if button == self.move_button:
            self.interaction_mode = "move"
        elif button == self.rotate_button:
            self.interaction_mode = "rotate"
        print(f"Interaction mode: {self.interaction_mode}")

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj == self.view:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.MouseMove and self.last_mouse_pos:
                delta = event.pos() - self.last_mouse_pos
                if self.interaction_mode == "rotate":
                    self.rotation_x += delta.y() * 0.3
                    self.rotation_y += delta.x() * 0.3
                    self.transform.setRotationX(self.rotation_x)
                    self.transform.setRotationY(self.rotation_y)
                elif self.interaction_mode == "move":
                    dx = delta.x() * 0.01 * self.camera_distance
                    dy = -delta.y() * 0.01 * self.camera_distance
                    self.translation += QVector3D(dx, dy, 0)
                    self.transform.setTranslation(-self.centroid + self.translation)
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.last_mouse_pos = None
                return True
            elif event.type() == QEvent.Wheel:
                self.camera_distance += event.angleDelta().y() * -0.05
                self.camera_distance = max(10.0, min(self.camera_distance, 1000.0))  # Extended range
                self.camera.setPosition(QVector3D(0, 0, self.camera_distance))
                return True
        return super().eventFilter(obj, event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Path to your .obj file
    obj_file = "/Users/francoisramon/Desktop/These/neuroguessr/code/temp_mri_mesh.obj"
    if not os.path.exists(obj_file):
        print(f"Error: {obj_file} not found")
        sys.exit(1)
    
    viewer = MeshViewer(obj_file)
    viewer.resize(800, 600)
    viewer.show()
    
    sys.exit(app.exec_())