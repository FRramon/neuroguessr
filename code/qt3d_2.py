import sys
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D, QColor
from PyQt5.Qt3DCore import QEntity, QTransform
from PyQt5.Qt3DExtras import Qt3DWindow, QOrbitCameraController, QSphereMesh, QPhongMaterial
from PyQt5.Qt3DRender import QGeometryRenderer, QGeometry, QAttribute, QBuffer, QCamera, QPointLight
from scipy.ndimage import gaussian_filter
from skimage import measure
import open3d as o3d
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class MRIMeshGeometry(QGeometry):
    def __init__(self, vertices, faces, normals, parent=None):
        super().__init__(parent)

        vertices = vertices.astype(np.float32)
        normals = normals.astype(np.float32)
        indices = faces.astype(np.uint32).flatten()

        # Vertex buffer
        vertex_data = vertices.tobytes()
        vertex_buffer = QBuffer(self)
        vertex_buffer.setData(vertex_data)
        vertex_attr = QAttribute(self)
        vertex_attr.setName(QAttribute.defaultPositionAttributeName())
        vertex_attr.setAttributeType(QAttribute.VertexAttribute)
        vertex_attr.setVertexBaseType(QAttribute.Float)
        vertex_attr.setVertexSize(3)
        vertex_attr.setByteOffset(0)
        vertex_attr.setByteStride(3 * 4)
        vertex_attr.setCount(len(vertices))
        vertex_attr.setBuffer(vertex_buffer)
        self.addAttribute(vertex_attr)

        # Normal buffer
        normal_data = normals.tobytes()
        normal_buffer = QBuffer(self)
        normal_buffer.setData(normal_data)
        normal_attr = QAttribute(self)
        normal_attr.setName(QAttribute.defaultNormalAttributeName())
        normal_attr.setAttributeType(QAttribute.VertexAttribute)
        normal_attr.setVertexBaseType(QAttribute.Float)
        normal_attr.setVertexSize(3)
        normal_attr.setByteOffset(0)
        normal_attr.setByteStride(3 * 4)
        normal_attr.setCount(len(normals))
        normal_attr.setBuffer(normal_buffer)
        self.addAttribute(normal_attr)

        # Index buffer
        index_data = indices.tobytes()
        index_buffer = QBuffer(self)
        index_buffer.setData(index_data)
        index_attr = QAttribute(self)
        index_attr.setAttributeType(QAttribute.IndexAttribute)
        index_attr.setVertexBaseType(QAttribute.UnsignedInt)
        index_attr.setByteOffset(0)
        index_attr.setByteStride(0)
        index_attr.setCount(len(indices))
        index_attr.setBuffer(index_buffer)
        self.addAttribute(index_attr)

class MRIViewer(Qt3DWindow):
    def __init__(self, nii_file):
        super().__init__()
        self.nii_file = nii_file
        self.brain_center = np.array([0, 0, 0])
        self.scale_factor = 1.0
        self.transform = QTransform()
        self.mode = 'rotate'
        self.crosshair_entity = None
        self.crosshair_pos = None
        self.last_mouse_pos = None

        # Black background
        self.defaultFrameGraph().setClearColor(QColor(0, 0, 0))

        # Root scene entity
        self.root_entity = QEntity()
        self.setRootEntity(self.root_entity)

        # Camera
        self.camera = self.camera()
        self.camera.setPosition(QVector3D(0, 0, 5))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

        # Orbit camera controller
        self.cam_controller = QOrbitCameraController(self.root_entity)
        self.cam_controller.setCamera(self.camera)

        # Light
        light_entity = QEntity(self.root_entity)
        light = QPointLight(light_entity)
        light.setColor(QColor(255, 255, 255))  # White
        light.setIntensity(1.0)
        light_transform = QTransform(light_entity)
        light_transform.setTranslation(QVector3D(1, 1, 1))
        light_entity.addComponent(light)
        light_entity.addComponent(light_transform)

        # Load and render mesh
        self.load_nifti()
    def add_axes(self):
        axis_length = 2.0
        axis_radius = 0.01

        def create_axis(color, translation, rotation_axis=None, angle=0):
            entity = QEntity(self.root_entity)
            mesh = QCylinderMesh()
            mesh.setRadius(axis_radius)
            mesh.setLength(axis_length)

            material = QPhongMaterial()
            material.setDiffuse(color)

            transform = QTransform()
            transform.setTranslation(translation)
            if rotation_axis:
                transform.setRotationX(angle if rotation_axis == 'x' else 0)
                transform.setRotationY(angle if rotation_axis == 'y' else 0)
                transform.setRotationZ(angle if rotation_axis == 'z' else 0)

            entity.addComponent(mesh)
            entity.addComponent(material)
            entity.addComponent(transform)

        # X-axis (red)
        create_axis(QColor(255, 0, 0), QVector3D(axis_length / 2, 0, 0), 'z', 90)
        # Y-axis (green)
        create_axis(QColor(0, 255, 0), QVector3D(0, axis_length / 2, 0))
        # Z-axis (blue)
        create_axis(QColor(0, 0, 255), QVector3D(0, 0, axis_length / 2), 'x', 90)

    def load_nifti(self):
        img = nib.load(self.nii_file)
        data = img.get_fdata()
        logging.info(f"Data shape: {data.shape}, min: {data.min()}, max: {data.max()}")

        data_min, data_max = data.min(), data.max()
        if data_max == data_min:
            logging.error("Data has no variation (min == max). Cannot normalize.")
            return
        data = (data - data_min) / (data_max - data_min)
        data = gaussian_filter(data, sigma=1)

        threshold = np.percentile(data, 75)
        logging.info(f"Using threshold: {threshold}")

        try:
            vertices, faces, normals, _ = measure.marching_cubes(data, threshold)
            logging.info(f"Generated {len(vertices)} vertices and {len(faces)} faces")

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.vertex_normals)
            logging.info(f"Simplified to {len(vertices)} vertices and {len(faces)} faces")

            dims = data.shape
            self.brain_center = np.array(dims) / 2
            vertices -= self.brain_center
            max_extent = np.max(np.abs(vertices))
            if max_extent == 0:
                logging.error("Max extent is zero. Cannot scale vertices.")
                return
            self.scale_factor = 2.0 / max_extent
            vertices *= self.scale_factor
            self.brain_center *= self.scale_factor

            geometry = MRIMeshGeometry(vertices, faces, normals, self.root_entity)

            mesh_renderer = QGeometryRenderer(self.root_entity)
            mesh_renderer.setGeometry(geometry)
            mesh_renderer.setPrimitiveType(QGeometryRenderer.Triangles)

            material = QPhongMaterial(self.root_entity)
            material.setAmbient(QColor(204, 204, 204))  # Light gray
            material.setDiffuse(QColor(204, 204, 204, 255))  # Fully opaque (alpha=255)

            brain_entity = QEntity(self.root_entity)
            brain_entity.addComponent(mesh_renderer)
            brain_entity.addComponent(material)
            brain_entity.addComponent(self.transform)

            self.add_axes()

        except Exception as e:
            logging.error(f"Error in marching cubes: {e}")

    def set_crosshair(self, x, y, z):
        pos = np.array([x, y, z]) - self.brain_center
        pos *= self.scale_factor

        if self.crosshair_entity:
            self.crosshair_entity.setParent(None)
            self.crosshair_entity = None

        self.crosshair_entity = QEntity(self.root_entity)
        sphere = QSphereMesh(self.crosshair_entity)
        sphere.setRadius(0.05)
        material = QPhongMaterial(self.crosshair_entity)
        material.setAmbient(QColor(255, 0, 0))  # Red
        transform = QTransform(self.crosshair_entity)
        transform.setTranslation(QVector3D(pos[0], pos[1], pos[2]))
        self.crosshair_entity.addComponent(sphere)
        self.crosshair_entity.addComponent(material)
        self.crosshair_entity.addComponent(transform)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            center = self.brain_center / self.scale_factor
            self.set_crosshair(center[0], center[1], center[2])

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            if self.mode == 'rotate':
                self.transform.setRotationX(self.transform.rotationX() + dy * 0.5)
                self.transform.setRotationY(self.transform.rotationY() + dx * 0.5)
            elif self.mode == 'move':
                current = self.transform.translation()
                self.transform.setTranslation(QVector3D(
                    current.x() + dx * 0.01,
                    current.y() - dy * 0.01,
                    current.z()
                ))
            self.last_mouse_pos = event.pos()

    def set_mode(self, mode):
        self.mode = mode
        self.cam_controller.setEnabled(self.mode == 'rotate')

class MainWindow(QMainWindow):
    def __init__(self, nii_file):
        super().__init__()
        self.setWindowTitle("3D MRI Viewer")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.viewer = MRIViewer(nii_file)
        container = self.createWindowContainer(self.viewer)
        layout.addWidget(container)

        button_layout = QHBoxLayout()
        move_button = QPushButton("Move")
        rotate_button = QPushButton("Rotate")
        move_button.clicked.connect(lambda: self.viewer.set_mode('move'))
        rotate_button.clicked.connect(lambda: self.viewer.set_mode('rotate'))
        button_layout.addWidget(move_button)
        button_layout.addWidget(rotate_button)
        layout.addLayout(button_layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    nii_file = "/Users/francoisramon/Desktop/These/neuroguessr/data/MNI_template_1mm_stride.nii.gz"  # Replace with your own file
    window = MainWindow(nii_file)
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec_())
