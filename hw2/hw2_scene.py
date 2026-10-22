import json
import numpy as np
import math
import os
import trimesh

class TriangleMesh:
    """
        A TriangleMesh contains the following:
        vertices -> [N, 3] float numpy array
        faces -> [M, 3] int numpy array
        vertex_colors -> [N, 3] float numpy array
        vertex_normals -> [N, 3] float numpy array
        vertex_uvs -> [N, 2] float numpy array
        model_matrix -> [4, 4] numpy array
        material -> {'Kd': (float, float, float),
                      'texture': string,
                      'Ks': (float, float, float)
                      'exponent': float}
    """
    def __init__(self,
                 vertices,
                 faces,
                 vertex_colors,
                 vertex_normals,
                 vertex_uvs,
                 model_matrix,
                 transform_keyframes,
                 material):
        self.vertices = vertices
        self.faces = faces
        self.vertex_colors = vertex_colors
        self.vertex_normals = vertex_normals
        self.vertex_uvs = vertex_uvs
        self.model_matrix = model_matrix
        self.transform_keyframes = transform_keyframes
        self.material = material

    def __repr__(self):
        return (
            f"TriangleMesh(\n"
            f"  vertices={self.vertices},\n"
            f"  faces={self.faces},\n"
            f"  vertex_colors={self.vertex_colors},\n"
            f"  vertex_normals={self.vertex_normals},\n"
            f"  vertex_uvs={self.vertex_uvs},\n"
            f"  model_matrix={self.model_matrix}\n"
            f"  transform_keyframes={self.transform_keyframes}\n"
            f"  material={self.material}\n"
            f")"
        )

class Camera:
    """
        A Camera contains the following:
        cam_to_world -> [4, 4] numpy array
        transform_keyframes -> [Transform]
        Transform ->  {'time': float,
                       'transform': [{'scale'|'translate'|'rotate_x'|'rotate_y'|'rotate_z'|'lookat': transformation values}]}
        resolution -> (int, int)
        s -> float
        z_near -> float
    """
    def __init__(self,
                 cam_to_world,
                 transform_keyframes,
                 resolution,
                 s,
                 z_near):
        self.cam_to_world = cam_to_world
        self.transform_keyframes = transform_keyframes
        self.resolution = resolution
        self.s = s
        self.z_near = z_near

    def __repr__(self):
        return (
            f"Camera(\n"
            f"  cam_to_world={self.cam_to_world},\n"
            f"  transform_keyframes={self.transform_keyframes},\n"
            f"  resolution={self.resolution},\n"
            f"  s={self.s},\n"
            f"  z_near={self.z_near}\n"
            f")"
        )

class Scene:
    """
        A Scene contains the following:
        camera -> Camera
        background -> (float, float, float)
        meshes -> [TriangleMesh]
        duration -> float (for HW 2.4)
        light_dir -> (float, float, float) (for HW2.7)
        light_color -> (float, float, float) (for HW2.7)
    """
    def __init__(self,
                 camera,
                 background,
                 meshes,
                 duration,
                 light_dir,
                 light_color):
        self.camera = camera
        self.background = background
        self.meshes = meshes
        self.duration = duration
        self.light_dir = light_dir
        self.light_color = light_color

    def __repr__(self):
        return (
            f"Scene(\n"
            f"  camera={self.camera},\n"
            f"  background={self.background},\n"
            f"  meshes={self.meshes},\n"
            f"  duration={self.duration}\n"
            f"  light_dir={self.light_dir}\n"
            f"  light_color={self.light_color}\n"
            f")"
        )

def parse_triangle_mesh(filename: str) -> TriangleMesh:
    if filename.lower().endswith('.ply'):
        mesh = trimesh.load_mesh(filename, process=False)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        # The "mesh.vertex_normals" call 
        # computes normals if they aren't stored in the PLY
        vertex_normals = np.asarray(
            mesh.vertex_normals, dtype=np.float32)

        uv = getattr(mesh.visual, 'uv', None)
        if uv is not None:
            vertex_uvs = np.asarray(uv, dtype=np.float32)
        else:
            # if there's no UVs we set everything to 00
            vertex_uvs = np.zeros(
                (len(vertices), 2), dtype=np.float32)

        if mesh.visual.kind == 'vertex':
            vertex_colors = np.asarray(
                mesh.visual.vertex_colors[:, :3],
                dtype=np.float32) / 255.0
        else:
            vertex_colors = np.ones_like(vertices)

    else:
        with open(filename, 'r') as f:
            mesh = json.load(f)

        vertices = np.array(mesh['vertices'], dtype=np.float32)
        faces = np.array(mesh['faces'], dtype=np.int32)
        vertex_colors = np.array(
            mesh['vertex_colors'], dtype=np.float32)

        if 'vertex_normals' in mesh:
            vertex_normals = np.array(
                mesh['vertex_normals'], dtype=np.float32)
        else:
            # Generate smooth vertex normals.
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False)
            vertex_normals = np.asarray(
                tri_mesh.vertex_normals, dtype=np.float32)

        vertex_uvs = np.array(
            mesh.get('vertex_uvs',
                     np.zeros((len(vertices), 2))),
            dtype=np.float32)

    return TriangleMesh(
        vertices, faces, vertex_colors,
        vertex_normals, vertex_uvs,
        model_matrix=np.eye(4, dtype=np.float32),
        transform_keyframes=[],
        material={})

def compose_transformations(transformations):
    """
        Given a list of transformations, compose them into a matrix.
    """
    F = np.eye(4, dtype=np.float32)
    T = np.eye(3, dtype=np.float32)

    for transform in transformations:
        # TODO: your code here
        for key, value in transform.items():
            match key:
                case "scale":
                    T = np.array([[value[0], 0, 0, 0], 
                                    [0, value[1], 0, 0], 
                                    [0, 0, value[2], 0],
                                    [0, 0, 0, 1]])
                case "translate":
                    T = np.array([[1, 0, 0, value[0]], 
                                    [0, 1, 0, value[1]], 
                                    [0, 0, 1, value[2]],
                                    [0, 0, 0, 1]])
                case "rotate_x":
                    angle = math.radians(value[0])
                    T = np.array([[1, 0, 0, 0], 
                                    [0, math.cos(angle), -math.sin(angle), 0], 
                                    [0, math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 0, 1]])
                case "rotate_y":
                    angle = math.radians(value[0])
                    T = np.array([[math.cos(angle), 0, math.sin(angle), 0], 
                                    [0, 1, 0, 0], 
                                    [-math.sin(angle), 0, math.cos(angle), 0],
                                    [0, 0, 0, 1]])
                case "rotate_z":
                    angle = math.radians(value[0])
                    T = np.array([[math.cos(angle), -math.sin(angle), 0, 0], 
                                    [math.sin(angle), math.cos(angle), 0, 0], 
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
                case "lookat":
                    p = value["position"]
                    t = value["target"]
                    u = value["up"]

                    sub_t_p = [x - y for x, y in zip(t, p)]

                    if np.linalg.norm(sub_t_p) == 0:
                        d = (sub_t_p)
                    else:
                        d = (sub_t_p)/(np.linalg.norm(sub_t_p))

                    if np.linalg.norm(np.cross(d, u)) == 0:
                        r = np.cross(d, u)
                    else:
                        r = (np.cross(d, u))/(np.linalg.norm(np.cross(d, u)))

                    u_prime = np.cross(r, d)

                    T = np.array([[r[0], u_prime[0], -d[0], p[0]],
                                    [r[1], u_prime[1], -d[1], p[1]],
                                    [r[2], u_prime[2], -d[2], p[2]],
                                    [0, 0, 0, 1]])

            F = T @ F

    return F

def parse_scene(filename : str) -> Scene:
    scene_dir = os.path.dirname(filename)
    with open(filename, 'r') as f:
        scene = json.load(f)
        duration = 1.0
        if 'duration' in scene:
            duration = scene['duration']
        camera = scene['camera']
        cam_to_world = np.eye(4, dtype=np.float32)
        if 'transform' in camera:
            cam_to_world = compose_transformations(camera['transform'])
        transform_keyframes = []
        if 'transform_keyframes' in camera:
            transform_keyframes = camera['transform_keyframes']
        camera = Camera(cam_to_world,
                        transform_keyframes,
                        camera.get('resolution', (640, 480)),
                        camera.get('s', 1.0),
                        camera.get('z_near', 1e-3))
        background = scene['background']
        meshes = []
        for obj in scene['objects']:
            mesh_filename = os.path.join(scene_dir, obj['filename'])
            mesh = parse_triangle_mesh(mesh_filename)
            if 'transform' in obj:
                mesh.model_matrix = compose_transformations(obj['transform'])
            if 'transform_keyframes' in obj:
                mesh.transform_keyframes = obj['transform_keyframes']
            # Use dictionary merging to set default material parameters
            mesh.material = {
                'Kd': [0.5, 0.5, 0.5],
                'Ks': [0.5, 0.5, 0.5],
                'exponent': 32.0,
                **obj.get('material', {})
            }
            meshes.append(mesh)

        light_dir = (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
        light_color = (1.0, 1.0, 1.0)
        if 'light' in scene:
            light = scene['light']
            if 'light_dir' in light:
                light_dir = light['light_dir']
                # normalize to unit direction
                light_dir = tuple((np.asarray(light_dir) / np.linalg.norm(light_dir)).tolist())
            if 'light_color' in light:
                light_color = light['light_color']

    return Scene(camera, background, meshes, duration, light_dir, light_color)
