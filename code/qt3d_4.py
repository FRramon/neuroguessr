import sys
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QUrl, QByteArray
from PyQt5.QtGui import QVector3D, QColor
from PyQt5.Qt3DCore import QEntity, QTransform
from PyQt5.Qt3DExtras import Qt3DWindow, QOrbitCameraController, QSphereMesh, QPhongMaterial
from PyQt5.Qt3DRender import QGeometryRenderer, QGeometry, QAttribute, QBuffer, QCamera, QPointLight
from PyQt5.Qt3DExtras import Qt3DWindow, QOrbitCameraController, QPhongMaterial, QForwardRenderer, QSphereGeometry
from PyQt5.Qt3DRender import QMesh, QCamera, QRenderStateSet, QDepthTest, QGeometryRenderer, QLineWidth
from PyQt5.QtGui import QVector3D, QColor
import uuid
import logging
import os

# Set up logging to file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifti_to_mesh_gui.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def nifti_to_obj(nifti_path, output_obj_path, threshold=1, smooth_iterations=2):
    logging.info(f"Converting NIfTI {nifti_path} to OBJ {output_obj_path}")
    try:
        # Load NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata().astype(np.int32)
        logging.debug(f"NIfTI data shape: {data.shape}, unique values: {np.unique(data)}")
        
        # Threshold: set values >= threshold to 1, others to 0
        binary_data = (data >= threshold).astype(np.uint8)
        logging.debug(f"After thresholding (>= {threshold}): {np.sum(binary_data)} voxels are 1")
        
        # Generate mesh using marching cubes
        verts, faces, _, _ = measure.marching_cubes(binary_data, level=0.5)
        logging.debug(f"Mesh generated: {verts.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Smooth the mesh using Laplacian smoothing
        for _ in range(smooth_iterations):
            mesh = trimesh.smoothing.filter_laplacian(mesh)
        logging.debug("Mesh smoothed")
        
        # Clean the mesh
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
        logging.debug(f"Mesh cleaned: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        # Apply affine transformation to vertices
        affine = img.affine
        verts_transformed = trimesh.transform_points(mesh.vertices, affine)
        mesh.vertices = verts_transformed
        
        # Export to OBJ
        mesh.export(output_obj_path)
        logging.info(f"OBJ file saved to {output_obj_path}")
        return output_obj_path, img.affine, data.shape
    except Exception as e:
        logging.error(f"Error in nifti_to_obj: {str(e)}")
        raise

class MeshViewer(QMainWindow):
    def __init__(self, obj_path, crosshair_coords, affine):
        super().__init__()
        self.setWindowTitle("3D Atlas Mesh Viewer")
        self.resize(800, 600)
        
        logging.info("Initializing MeshViewer")
        try:
            # Create 3D window
            self.view = Qt3DWindow()
            self.container = self.createWindowContainer(self.view)
            
            # Set up layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            layout.addWidget(self.container)
            
            # Create scene
            self.root_entity = QEntity()
            self.setup_scene(obj_path, crosshair_coords, affine)
            
            # Set up view
            self.view.setRootEntity(self.root_entity)
            logging.debug("Root entity set")
        except Exception as e:
            logging.error(f"Error in MeshViewer.__init__: {str(e)}")
            raise
        
    def setup_scene(self, obj_path, crosshair_coords, affine):
        logging.info(f"Setting up scene with OBJ path: {obj_path}, crosshair at {crosshair_coords}")
        try:
            # Verify OBJ file exists
            if not os.path.exists(obj_path):
                logging.error(f"OBJ file not found: {obj_path}")
                raise FileNotFoundError(f"OBJ file not found: {obj_path}")
            
            # Camera
            camera = self.view.camera()
            camera.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 1000.0)
            camera.setPosition(QVector3D(0, 0, 150))  # Position farther away
            camera.setViewCenter(QVector3D(0, 0, 0))

            cam_controller = QOrbitCameraController(self.root_entity)
            cam_controller.setCamera(camera)
            cam_controller.setLinearSpeed(100.0)  # Increase movement speed
            cam_controller.setLookSpeed(180.0)    # Increase rotation speed

            

            # Mesh
            mesh = QMesh()
            mesh.setSource(QUrl.fromLocalFile(obj_path))
            logging.debug(f"Mesh source set to: {QUrl.fromLocalFile(obj_path).toString()}")
            
            # Material
            material = QPhongMaterial()
            material.setAmbient(Qt.blue)
            
            # Transform
            transform = QTransform()
            #transform.setScale3D(QVector3D(0.2, 0.2, 0.2))  # Scale down to 20% of original size


            # Mesh entity
            mesh_entity = QEntity(self.root_entity)
            mesh_entity.addComponent(mesh)
            mesh_entity.addComponent(material)
            mesh_entity.addComponent(transform)
            logging.debug("Mesh entity configured")
            
            # Crosshair: Transform NIfTI coordinates to world space
            crosshair_coords = np.array([crosshair_coords], dtype=np.float32)
            world_coords = trimesh.transform_points(crosshair_coords, affine)[0]
            logging.debug(f"Crosshair world coordinates: {world_coords}")
            
            # Create crosshair as three perpendicular lines
            crosshair_entity = QEntity(self.root_entity)
            crosshair_size = 10.0  # Increased length of each crosshair line
            
            # Define vertices for three lines (x, y, z axes)
            vertices = np.array([
                # X-axis line
                [world_coords[0] - crosshair_size, world_coords[1], world_coords[2]],
                [world_coords[0] + crosshair_size, world_coords[1], world_coords[2]],
                # Y-axis line
                [world_coords[0], world_coords[1] - crosshair_size, world_coords[2]],
                [world_coords[0], world_coords[1] + crosshair_size, world_coords[2]],
                # Z-axis line
                [world_coords[0], world_coords[1], world_coords[2] - crosshair_size],
                [world_coords[0], world_coords[1], world_coords[2] + crosshair_size]
            ], dtype=np.float32)
            
            # Define line indices
            indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
            
            # Create geometry for lines
            geometry = QGeometry(crosshair_entity)
            
            # Vertex buffer
            vertex_data = vertices.tobytes()
            vertex_buffer = QBuffer(crosshair_entity)
            vertex_buffer.setData(QByteArray(vertex_data))
            
            vertex_attribute = QAttribute(geometry)
            vertex_attribute.setName(QAttribute.defaultPositionAttributeName())
            vertex_attribute.setVertexBaseType(QAttribute.Float)
            vertex_attribute.setVertexSize(3)
            vertex_attribute.setAttributeType(QAttribute.VertexAttribute)
            vertex_attribute.setBuffer(vertex_buffer)
            vertex_attribute.setByteStride(3 * 4)  # 3 floats * 4 bytes
            vertex_attribute.setCount(vertices.shape[0])
            geometry.addAttribute(vertex_attribute)
            
            # Index buffer
            index_data = indices.tobytes()
            index_buffer = QBuffer(crosshair_entity)
            index_buffer.setData(QByteArray(index_data))
            
            index_attribute = QAttribute(geometry)
            index_attribute.setAttributeType(QAttribute.IndexAttribute)
            index_attribute.setVertexBaseType(QAttribute.UnsignedInt)
            index_attribute.setBuffer(index_buffer)
            index_attribute.setCount(indices.shape[0])
            geometry.addAttribute(index_attribute)
            
            # Geometry renderer for lines
            geom_renderer = QGeometryRenderer(crosshair_entity)
            geom_renderer.setGeometry(geometry)
            geom_renderer.setPrimitiveType(QGeometryRenderer.Lines)
            
            # Material for crosshair lines (red)
            crosshair_material = QPhongMaterial()
            crosshair_material.setAmbient(QColor(255, 0, 0))
            
            # Add components to crosshair entity
            crosshair_entity.addComponent(geom_renderer)
            crosshair_entity.addComponent(crosshair_material)
            
            # Add sphere at crosshair center
            sphere_entity = QEntity(self.root_entity)
            sphere_geometry = QSphereGeometry()
            sphere_geometry.setRadius(1.0)  # Small sphere
            sphere_renderer = QGeometryRenderer(sphere_entity)
            sphere_renderer.setGeometry(sphere_geometry)
            sphere_material = QPhongMaterial()
            sphere_material.setAmbient(QColor(255, 0, 0))
            sphere_transform = QTransform()
            sphere_transform.setTranslation(QVector3D(world_coords[0], world_coords[1], world_coords[2]))
            sphere_entity.addComponent(sphere_renderer)
            sphere_entity.addComponent(sphere_material)
            sphere_entity.addComponent(sphere_transform)
            logging.debug("Crosshair (lines and sphere) configured")
            
            # Frame graph with ForwardRenderer
            forward_renderer = QForwardRenderer()
            forward_renderer.setCamera(camera)
            forward_renderer.setClearColor(Qt.gray)
            
            # Render state set with depth test for mesh
            render_state_set = QRenderStateSet()
            depth_test = QDepthTest()
            depth_test.setDepthFunction(QDepthTest.Less)
            render_state_set.addRenderState(depth_test)
            
            # Increase line width for crosshair visibility
            line_width = QLineWidth()
            line_width.setValue(3.0)
            render_state_set.addRenderState(line_width)
            
            # Set up frame graph hierarchy
            render_state_set.setParent(self.root_entity)
            forward_renderer.setParent(render_state_set)
            self.view.setActiveFrameGraph(forward_renderer)
            logging.debug("Frame graph configured")
        except Exception as e:
            logging.error(f"Error in setup_scene: {str(e)}")
            raise

def main():
    logging.info("Starting main function")
    try:
        # Convert NIfTI to OBJ
        input_nifti = "/Users/francoisramon/Desktop/These/neuroguessr/data/xtract.nii.gz"
        output_obj = "/Users/francoisramon/Desktop/These/neuroguessr/data/atlas_mesh.obj"
        obj_path, affine, data_shape = nifti_to_obj(input_nifti, output_obj)
        
        # Set crosshair at the center of the NIfTI data
        crosshair_coords = [47, 77, 114]
        crosshair_coords = [data_shape[0] // 2, data_shape[1] // 2, data_shape[2] // 2]

        logging.info(f"Crosshair coordinates set to NIfTI center: {crosshair_coords}")
        
        # Create and show GUI
        app = QApplication(sys.argv)
        viewer = MeshViewer(obj_path, crosshair_coords, affine)
        viewer.show()
        logging.info("GUI displayed")
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()